# 02. Task suite: rack_cell

Level 1: a Franka Panda in front of a small section of a server rack, with 3 data center micro-tasks. One reads the panel, one slides a 1U box into a slot, one inserts a keyed plug into a port. Each has a privileged checker and a seed-deterministic scene.

## Geometry (real-world references, scaled where noted)

- 19-inch rack: panel width 482.6 mm, 1U = 44.45 mm. Model 4U of rack height with 2 shelf slots and 1 patch panel row.
- Panda payload is 3 kg and the gripper opens to 80 mm. Scale the 1U box so the arm can carry it: about 200 x 150 x 44.45 mm, 1.0 kg. Grip it across the 44.45 mm height or by a handle. Record the scale in the report.
- Shelf rails: two L-rails per slot. The box rests on them when seated.
- Patch panel: 12 rectangular ports, 12 x 9 mm opening, 0.5 mm total clearance to the plug, pitch about 15 mm. A 6 mm port depth limit before a stop face is fine, but T3 needs at least 8 mm insertion, so make the port 12 mm deep.
- LED above each port: small emissive geom with rgba green (0,1,0), amber (1,0.6,0), red (1,0,0) or off (dark grey). Off must be distinguishable from the panel color.
- Label strip with port numbers 1 to 12, rendered as a texture so it is readable in the front camera.
- Parts tray at a known nominal location holding the keyed plug (11.5 x 8.5 x 20 mm, with a key so it fits one way) and the 1U box.

## Cameras (640 x 480)

- `overhead`: looking down at tray and rack front edge
- `front`: facing the panel, about 50 cm away, can read labels and LEDs
- `wrist`: on the Panda hand, looking along the approach axis

Save camera intrinsics and extrinsics in each Observation.

## Randomization per seed

- Rack pose: plus or minus 2 cm in x and y, plus or minus 3 deg yaw around the nominal
- Lighting: 2 to 4 lights, intensity and direction jitter
- LED colors per port, from the 4 states
- Occupied ports: 0 to 6 ports already hold a plug
- Target port (always unoccupied for T3) and target slot
- Tray part pose jitter: plus or minus 5 mm, plus or minus 10 deg yaw

Same seed must give the same scene bit for bit. Test it.

## Tasks

| Task | Goal | Pass rule | Score |
|---|---|---|---|
| T1 inspect | Output JSON `{"1": {"occupied": bool, "led": "green|amber|red|off"}, ...}` for all 12 ports | Every field right | Fraction of the 24 fields right |
| T2 rack_slot | Put the 1U box in slot n | Position error at most 3 mm, yaw error at most 2 deg, box resting on both rails, gripper released | 1 if passed, else partial from errors |
| T3 plug_insert | Insert the plug into port n | Depth at least 8 mm, lateral error at most 0.5 mm, plug released and still seated after 1 s | 1 if passed, else partial |

Checkers read privileged state (body poses, contacts) and return a Verdict with metrics and one evidence image. Assign `failure_class` from the metrics: wrong port or missed localization is `perception`, dropped part is `grasp`, lateral error or jam is `alignment`, a robot that never ends or keeps moving after success is `stopping`, budget hit is `timeout`, bounds or force breach is `safety`.

## Oracles and sanity programs

- Oracle programs for T2 and T3 that read privileged state. They prove the task is solvable with the scene's contact settings. Target: at least 95% over 50 dev seeds.
- A do-nothing program scores 0 on every task.
- One mp4 per task to `LOOPKIT_DATA/videos/`.

## Contact tuning notes

Sub-millimeter clearance is hard in MuJoCo. Start with `noslip_iterations`, a smaller timestep (1 to 2 ms), `solref` and `solimp` tuned on the plug and port, a chamfer on the port mouth, and box or mesh collision for the plug. If T3 is not solvable after 3 tuning attempts, loosen clearance to 1.0 mm, record it in the report and move on.
