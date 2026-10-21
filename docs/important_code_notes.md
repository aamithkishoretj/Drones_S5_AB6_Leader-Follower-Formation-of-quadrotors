| Code location | What it does |
|---|---|
| `simulators/leader_follower.py:156-161` | Records the current state, applied command, and actual next state for training. |
| `scripts/train_data_driven.py:84-86` | Passes the collected training data into model fitting. |
| `dq_control/learned_control.py:75-95` | Learns the response-model values A, B, and c using ridge regression. |
| `dq_control/learned_control.py:48-49` | Predicts the next movement state using the learned model. |
| `simulators/leader_follower.py:43-48` | Loads the saved model and selects the learned controller. |
| `dq_control/learned_control.py:29-37` | Calculates translation and rotation error using dual quaternions. |
| `dq_control/learned_control.py:117-121` | Builds predictions over the control horizon using the learned model. |
| `dq_control/learned_control.py:126-147` | Uses the current state and pose error to optimize a bounded flight command. |
| `simulators/leader_follower.py:126-135` | Sends the leader's live position and orientation to the follower's path tracker. |
| `dq_control/path_following.py:19-25` | Records the leader's current-flight path and selects a distance behind it. |
| `dq_control/path_following.py:28-31` | Interpolates the follower's target position and orientation along that path. |
| `simulators/leader_follower.py:100-101` | Requests trail drawing using the drones' actual positions. |
| `scripts/train_data_driven.py:95-97` | Checks prediction accuracy on a separate validation flight. |
| `scripts/train_data_driven.py:103-118` | Runs analytic and learned controller comparisons and saves evaluation results. |
