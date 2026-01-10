Put the files inside libero_90 directory. Rename properly

```
Filepath example: /home/ns1254/openvla/LIBERO/libero/datasets/libero_90
Files inside (one or more): LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate_demo.hdf5

cd openvla
python experiments/robot/libero/regenerate_libero_dataset.py \
            --libero_task_suite libero_90 \
            --libero_raw_data_dir ./LIBERO/libero/datasets/libero_90 \
            --libero_target_dir ./LIBERO/libero/datasets/libero_90_no_noops
```
