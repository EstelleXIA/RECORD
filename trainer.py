import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from utils.plot_utils import classification_report
from monai.data import decollate_batch
import json
from monai.networks.blocks import Warp
import SimpleITK as sitk


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, model_inferer=None, progression_loss_func=None):
    assert progression_loss_func is not None
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    progression_losses = []
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        baseline = os.path.basename(batch_data["image_meta_dict"]["filename_or_obj"][0])
        followup = os.path.basename(batch_data["image_1_meta_dict"]["filename_or_obj"][0])
        pair_inputs = baseline != followup
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        warp_linear = Warp(mode="bilinear", padding_mode="border")
        if pair_inputs:
            data_1, target_1 = batch_data["image_1"].cuda(args.rank), batch_data["label_1"].cuda(args.rank)
            displ_path = os.path.join(args.transform_path, baseline.split(".")[0] + "_" + followup)
            displ = sitk.ReadImage(displ_path)
            torch_ddf = torch.from_numpy(np.float32(sitk.GetArrayFromImage(displ)).transpose(
                (2, 1, 0, 3))).unsqueeze(0).swapaxes(0, -1)[:, :, :, :, 0].unsqueeze(0).cuda(args.rank)
            torch_ddf_div = torch_ddf / torch.tensor([2.0, 2.0, 2.0]).view(1, -1, 1, 1, 1).cuda(args.rank)

        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            if model_inferer is None:
                logits = model(data)
            else:
                logits = model_inferer(data)
                if pair_inputs:
                    logits_1 = model_inferer(data_1)

            prob_0 = torch.softmax(logits, 1)
            loss = loss_func(prob_0, target)

            if pair_inputs:
                prob_1 = torch.softmax(logits_1, 1)
                prob_1 = warp_linear(prob_1, torch_ddf_div)
                loss_1 = loss_func(prob_1, target_1)
                progression_loss = progression_loss_func(prob_0, prob_1, target, target_1)
                loss = (loss + loss_1) / 2 + args.weight * progression_loss
                progression_losses.append(progression_loss.item())

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "progression_loss: {:.4f}".format(progression_loss.item() if pair_inputs else -1),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg, np.mean(progression_losses)


def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None,
              post_label=None, post_pred=None, progression_loss_func=None):
    assert progression_loss_func is not None
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    d = []
    val_progression_losses = []

    gt_ps, gt_labels, pred_ps = [], [], []

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            data_1, target_1 = batch_data["image_1"].cuda(args.rank), batch_data["label_1"].cuda(args.rank)
            baseline = os.path.basename(batch_data["image_meta_dict"]["filename_or_obj"][0])
            followup = os.path.basename(batch_data["image_1_meta_dict"]["filename_or_obj"][0])
            displ_path = os.path.join(args.transform_path, baseline.split(".")[0] + "_" + followup)
            displ = sitk.ReadImage(displ_path)
            torch_ddf = torch.from_numpy(np.float32(sitk.GetArrayFromImage(displ)).transpose(
                (2, 1, 0, 3))).unsqueeze(0).swapaxes(0, -1)[:, :, :, :, 0].unsqueeze(0).cuda(args.rank)
            torch_ddf_div = torch_ddf / torch.tensor([2.0, 2.0, 2.0]).view(1, -1, 1, 1, 1).cuda(args.rank)

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    warp_linear = Warp(mode="bilinear", padding_mode="border")
                    logits = model_inferer(data)
                    logits_1 = model_inferer(data_1)
                    prob_0 = torch.softmax(logits, 1)
                    prob_1 = torch.softmax(logits_1, 1)
                    prob_1 = warp_linear(prob_1, torch_ddf_div)

                    if args.val:
                        affine = batch_data["image_meta_dict"]["affine"][0].numpy()
                        affine[0, 0] = args.space_x
                        affine[1, 1] = args.space_y
                        affine[2, 2] = args.space_z
                        save_folder = os.path.join(args.logdir, "transformed_prediction")

                        def save_img(img, ref_img_name, save_path):
                            img_new = sitk.GetImageFromArray(img.transpose((2, 1, 0)))
                            ref_path = "data/mask/"
                            if os.path.exists(ref_path):
                                img_new.SetSpacing(sitk.ReadImage(os.path.join(ref_path, ref_img_name)).GetSpacing())
                                img_new.SetDirection(sitk.ReadImage(os.path.join(ref_path, ref_img_name)).GetDirection())
                                img_new.SetOrigin(sitk.ReadImage(os.path.join(ref_path, ref_img_name)).GetOrigin())
                            sitk.WriteImage(img_new, save_path)

                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)
                            os.makedirs(os.path.join(save_folder, "labels_continuous"))
                            os.makedirs(os.path.join(save_folder, "labels_discrete"))

                        # save baseline and follow-up predicted masks
                        name_0 = os.path.basename(batch_data["label_meta_dict"]["filename_or_obj"][0]).split(".")[0] \
                                 + "-" + os.path.basename(batch_data["label_1_meta_dict"]["filename_or_obj"][0])

                        save_img(img=prob_0.cpu().numpy()[0, 1],
                                 ref_img_name=os.path.basename(batch_data["label_meta_dict"]["filename_or_obj"][0]),
                                 save_path=os.path.join(save_folder, "labels_continuous", name_0))

                        save_img(img=prob_0.argmax(dim=1).cpu().numpy()[0].astype(np.uint8),
                                 ref_img_name=os.path.basename(batch_data["label_meta_dict"]["filename_or_obj"][0]),
                                 save_path=os.path.join(save_folder, "labels_discrete", name_0))

                        save_img(img=prob_1.cpu().numpy()[0, 1],
                                 ref_img_name=os.path.basename(batch_data["label_1_meta_dict"]["filename_or_obj"][0]),
                                 save_path=os.path.join(save_folder, "labels_continuous", os.path.basename(
                                           batch_data["label_1_meta_dict"]["filename_or_obj"][0])))

                        save_img(img=prob_1.argmax(dim=1).cpu().numpy()[0].astype(np.uint8),
                                 ref_img_name=os.path.basename(batch_data["label_1_meta_dict"]["filename_or_obj"][0]),
                                 save_path=os.path.join(save_folder, "labels_discrete", os.path.basename(
                                           batch_data["label_1_meta_dict"]["filename_or_obj"][0])))

                    val_progression_loss = progression_loss_func(prob_0, prob_1, target, target_1)

                    pred_ps.append(progression_loss_func.prob(prob_0[:, 1].sum((1, 2, 3)),
                                                              prob_1[:, 1].sum((1, 2, 3))))
                    gt_ps.append(progression_loss_func.prob(target[:, 0].sum((1, 2, 3)), target_1[:, 0].sum((1, 2, 3))))

                    gt_labels.append(torch.argmax(progression_loss_func.prob(target[:, 0].sum((1, 2, 3))
                                                                             , target_1[:, 0].sum((1, 2, 3))), dim=1))
                    val_progression_losses.append(val_progression_loss)
                else:
                    logits = model(data)

            if not prob_0.is_cuda:
                target = target.cpu()
                target_1 = target_1.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_1_list = decollate_batch(target_1)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_labels_1_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_1_list]
            val_outputs_list = decollate_batch(prob_0)
            val_outputs_1_list = decollate_batch(prob_1)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            val_output_1_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_1_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            acc_func.reset()
            acc_func(y_pred=val_output_1_convert, y=val_labels_1_convert)
            acc_1, not_nans_1 = acc_func.aggregate()
            acc_1 = acc_1.cuda(args.rank)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                if os.path.basename(batch_data["image_meta_dict"]["filename_or_obj"][0]) not in d:
                    d.append(os.path.basename(batch_data["image_meta_dict"]["filename_or_obj"][0]))
                    run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
                run_acc.update(acc_1.cpu().numpy(), n=not_nans_1.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
        gt_ps = torch.cat(gt_ps, 0)
        gt_labels = torch.cat(gt_labels, 0)
        pred_ps = torch.cat(pred_ps, 0)
        with open(os.path.join(args.logdir, f"record-report-epoch-{epoch}.json"), "w") as f:
            json.dump(classification_report(gt_labels, gt_ps, pred_ps,
                                            figure_name=os.path.join(args.logdir, f"record-epoch-{epoch}.png")), f)
    return run_acc.avg, torch.stack(val_progression_losses).mean()


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.module.state_dict() if args.distributed or isinstance(model, torch.nn.DataParallel) else model.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        acc_func,
        args,
        model_inferer=None,
        scheduler=None,
        start_epoch=0,
        post_label=None,
        post_pred=None,
        progression_loss_func=None
):
    assert progression_loss_func is not None
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    val_progression_metric_max = 1e2
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss, train_progression_metric = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args,
            model_inferer=model_inferer, progression_loss_func=progression_loss_func
        )
        # train_loss = 0
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "progression metric: {:.4f}".format(train_progression_metric),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            # model.load_state_dict(torch.load("pretrained_models/swin_unetr.f48_lits_beigene_fold0_0.7973.pt")["state_dict"])
            val_avg_acc, val_progression_metric = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
                progression_loss_func=progression_loss_func
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "progression metric", val_progression_metric,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_progression_metric < val_progression_metric_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_progression_metric_max, val_progression_metric))
                    val_progression_metric_max = val_progression_metric
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(model, epoch, args, best_acc=val_progression_metric_max,
                                        optimizer=optimizer, scheduler=scheduler)
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model,
                                epoch,
                                args,
                                best_acc=val_progression_metric_max,
                                filename="model_final.pt")
                print("Copying to model.pt new best model!!!!") if b_new_best else None
                save_checkpoint(model,
                                epoch,
                                args,
                                best_acc=val_progression_metric_max,
                                filename=f'model-dice_{val_avg_acc:.4f}-progression_{val_progression_metric:.6f}.pt')
                shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
