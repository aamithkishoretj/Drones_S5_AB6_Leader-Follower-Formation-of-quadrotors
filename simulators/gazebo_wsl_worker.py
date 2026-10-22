"""WSL-side command worker for the Windows desktop Gazebo backend."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback


def _state_packet(state):
    return {
        "position": state.position.tolist(),
        "attitude": state.attitude.as_array().tolist(),
        "velocity": state.velocity.tolist(),
        "angular_velocity": state.angular_velocity.tolist(),
        "motor_rpm": state.motor_rpm.tolist(),
    }


def main() -> int:
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    protocol_out = sys.stdout
    # Keep stdout exclusively for the line-delimited JSON protocol. Simulator
    # startup messages, dependency warnings, and diagnostics go to stderr.
    sys.stdout = sys.stderr
    backend = None
    from simulators.base import SimConfig, SimulationStopped, VehicleCommand
    from simulators.gazebo import GazeboBackend

    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request.get("op")
            if operation == "reset":
                if backend is not None:
                    backend.close()
                name = request.get("backend", "gazebo")
                config = SimConfig(simulator=name, **request["config"])
                if name == "ardupilot":
                    from simulators.ardupilot import ArduPilotBackend
                    backend = ArduPilotBackend(config)
                else:
                    backend = GazeboBackend(config)
                backend.reset(request["initial_xyzs"], request["initial_rpys"])
                result = [_state_packet(s) for s in backend.get_states()]
            elif operation == "get_states":
                result = [_state_packet(s) for s in backend.get_states()]
            elif operation == "apply_commands":
                if backend is None:
                    raise RuntimeError("Gazebo backend has not been reset")
                commands = [VehicleCommand(**row) for row in request["commands"]]
                backend.apply_commands(commands)
                result = None
            elif operation == "step":
                if backend is None:
                    raise RuntimeError("Gazebo backend has not been reset")
                result = [_state_packet(s) for s in backend.step()]
            elif operation == "draw_trails":
                backend.draw_trails(request["positions"])
                result = None
            elif operation == "close":
                if backend is not None:
                    backend.close()
                    backend = None
                print(json.dumps({"ok": True, "result": None}), file=protocol_out, flush=True)
                return 0
            else:
                raise ValueError(f"Unknown Gazebo worker operation: {operation!r}")
            response = {"ok": True, "result": result}
        except SimulationStopped as exc:
            response = {"ok": True, "stopped": True, "reason": str(exc)}
        except BaseException:
            response = {"ok": False, "error": traceback.format_exc()}
        print(json.dumps(response, separators=(",", ":")), file=protocol_out, flush=True)

    if backend is not None:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
