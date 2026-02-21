# Data Visualization Tool for Cajun

<!-- * Re-structure Cajun -->
<!--     - [.] Do not change the saved data names - to anyother format. But add to metadata -->
<!--     - [ ] Load everything from config file(all magic numbers) -->
<!--     - [ ] Automate data capture in eval script -->
<!--     - [.] Capture Observation & Action Spaces -->

## Initial Setup
```bash
sudo apt-get install -y \
  libffi-dev libbz2-dev libsqlite3-dev tk-dev liblzma-dev \
  libssl-dev libreadline-dev zlib1g-dev libncurses-dev \
  libgdbm-dev libnss3-dev uuid-dev libxrandr-dev \
  libxinerama-dev libxcursor-dev libxi-dev libgl1-mesa-dev \
  libglu1-mesa-dev libglew-dev libglfw3-dev libxext-dev \
  libx11-dev build-essential cmake ninja-build pkg-config

```

## RoadMap:
- [x] Fix adding new graphs
- [x] Make any data plotable with directory walk - eg: non-standard data like des_acc (or clipping bounds)
- [x] Add Matlab Like Probes on graph
- [x] Plot Any data?
- [ ] Verify if 3D force rendering is accurate
- [ ] Perturbation Rendering(Make Preloaded, but Auxlillary Data) --> Yasser
- [ ] Better (URDF/Data) path loading
- [ ] Make bounds loaded from config optional, --> Bounds not drawn in graphs
- [.] If constant data, but may be varaible --> (What to do?/How do save/load/render/plot?) --> Eg: Constant kp/kd

- [ ] Re-structure Cajun
    * Save lb/ub limits into config, not metadata
    * Load eveything from config when training or eval
    * Automate data capture, what to capture:
        - [ ] Observation/Action Space + (LB/UB)
        - [ ] Joint Torques + (LB/UB)
        - [ ] Joint Velocities + (LB/UB)
        - [ ] Base Pos (Pose -> Pos + Orn) (*No UB/LB)
        - [ ] Foot Contact States (*No UB/LB)
        - [ ] Foot Contact Forces (*No UB/LB)
        - [ ] Time (*No UB/LB)
        - [ ] * Constant Kp/Kds (May help with {HZD Impl} realtime) 
        - [ ] * Variable Kp/Kds (May help with {HZD Impl} realtime) 

