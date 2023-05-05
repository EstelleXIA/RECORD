python main.py --json_list=record_optim.json \
--data_dir=prep/ \
--val_every=1 \
--logdir=record_log/ \
--dif_map_path=data/dif_map/ \
--weight=0.55 \
--space_x=2 \
--space_y=2 \
--space_z=2 \
--out_channels=2 \
--batch_size=1 \
--max_epochs=1000 \
--save_checkpoint \
--infer_overlap=0.5 \
--probabilistic