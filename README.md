# 두산 로봇 모방학습 파이프라인 (LeRobot + SO101)

LeRobot SO101 리더/팔로워 텔레오프 구조로 두산 E0509을 데이터 수집 → ACT 학습 → 실로봇 추론까지 연결한 워크플로우입니다.

## 구성 파일
- `so_to_real_dsr_teleop.py`  
  - SO101 리더(조이스틱/가상팔)에서 퍼블리시되는 `/joint_states`를 받아 두산 로봇 `/dsr01/motion/move_joint` + DRL 그리퍼로 그대로 따라 하게 하는 ROS2 노드. 좌표/부호/오프셋 변환과 스트로크 양자화를 포함합니다.
- `act_doosan_bridge.py`  
  - LeRobot ACT 정책 체크포인트를 불러 두 대의 카메라(Top/Wrist)와 두산 조인트 상태를 관측 → 정책 행동을 SO101 포맷의 `JointState`로 `/joint_states`에 퍼블리시. 위 텔레오프 노드가 이를 받아 실제 로봇을 구동합니다. Rerun 시각화 옵션 포함.
- `lerobot_record_doosan.py` (원본 위치: `lerobot/src/lerobot/scripts/lerobot_record_doosan.py`)  
  - SO101 리더(teleop) ↔ 팔로워(두산) 구조로 RGB 스트림 + 조인트를 기록해 HF Datasets에 업로드하는 스크립트.

## 데이터 수집 (모방 학습용)
SO101 리더/팔로워를 USB 포트에 연결한 뒤:
```bash
python -m lerobot.scripts.lerobot_record_doosan \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{top: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 15, warmup_s: 3}, \
                      wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 15, warmup_s: 3}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_awesome_leader_arm \
    --dataset.repo_id=${HF_USER}/so101_test25 \
    --dataset.fps=15 \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=5 \
    --dataset.single_task="pick and place" \
    --display_data=true \
    --play_sounds=true
```
- `robot.port`/`teleop.port`를 실제 USB 포트로 변경하세요.
- 카메라 인덱스(8, 6)는 환경에 맞게 조정.

## 학습 (ACT 정책)
HF Datasets에 올린 데이터로 ACT 학습:
```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=bluephysi01/so101_test25 \
  --policy.type=act \
  --job_name=act_so101_test25 \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=${HF_USER}/act_so101_test25 \
  --steps=100000 \
  --wandb.enable=true
```
- 완료 후 `outputs/.../pretrained_model/` 폴더(또는 Hub에 푸시된 repo)를 inference에 사용합니다.

## 실로봇 추론(재현) 파이프라인
1. 두산 bringup 및 ROS2 환경 설정 후, 터미널 A에서 텔레오프 노드 실행:  
   ```bash
   ros2 run dsr_example2 so_to_real_dsr_teleop.py
   ```
2. 터미널 B에서 ACT → 두산 브리지 실행:  
   ```bash
   python act_doosan_bridge.py \
     --pretrained_path /home/bluephysi01/lerobot/outputs/train/2025-11-18/00-48-48_act_so101_test25/checkpoints/last/pretrained_model \
     --device cuda \
     --rate 30 \
     --top_camera_index 8 \
     --wrist_camera_index 4 \
     --max_steps 600 \
     --smoothing_alpha 0.3 \
     --action_scale 0.5 \
     --robot_id dsr01 \
     --display_data
   ```
3. 브리지가 `/joint_states`에 SO101 포맷 조인트를 퍼블리시 → 텔레오프 노드가 이를 받아 `/dsr01/motion/move_joint`와 DRL 그리퍼로 실제 로봇을 구동합니다.

## 주의 및 팁
- ROS_DOMAIN_ID, 네임스페이스가 다를 경우 `--robot_id`와 토픽 이름을 맞춰주세요.
- 카메라 인덱스/해상도는 `act_doosan_bridge.py` 옵션으로 조정 가능합니다.
- 그리퍼 스트로크는 0~700 양자화, 조인트 명령은 0.15s 주기. 명령 과출력을 막기 위해 허용 오차를 갖습니다.
- HF 업로드 전 `HF_USER` 환경변수 설정을 권장합니다.
