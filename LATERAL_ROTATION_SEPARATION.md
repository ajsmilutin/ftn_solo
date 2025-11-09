# Lateral and Rotation Separation Fix

## The Problem

Previously, lateral movement (Y) and rotation (Z) were **mixed together**:

```python
# OLD CODE (BROKEN):
if off_rot > 0.001:
    effective_off_y = off_rot  # ❌ Rotation overwrites lateral!
    use_rotation_pattern = True
else:
    effective_off_y = off_y
    use_rotation_pattern = False
```

**Issues:**
- Rotation would **overwrite** lateral movement
- Couldn't move laterally AND rotate at the same time
- Direction flags got confused

## The Fix

Now lateral and rotation are **completely separate**:

```python
# NEW CODE (FIXED):
use_rotation = off_rot > 0.001
use_lateral = off_y > 0.001

# Separate splines for each
front_y_lateral = np.array([0.1469-off_y, 0.1469, 0.1469+off_y])
back_y_lateral = np.array([0.1469+off_y, 0.1469, 0.1469-off_y])

front_y_rotation = np.array([0.1469-off_rot, 0.1469, 0.1469+off_rot])
back_y_rotation = np.array([0.1469+off_rot, 0.1469, 0.1469-off_rot])
```

## Three Movement Modes

### 1. Pure Lateral Movement
```python
if use_lateral and not use_rotation:
    # Only lateral offset (off_y)
    # Left/right movement
```

**Example:**
```
Command: linear.y = 0.02 (move left)
Result: Robot moves left, no rotation
```

### 2. Pure Rotation
```python
if use_rotation and not use_lateral:
    # Only rotation offset (off_rot)
    # Rotate in place
```

**Example:**
```
Command: angular.z = 0.015 (rotate CCW)
Result: Robot rotates, no lateral movement
```

### 3. Combined Movement
```python
if use_rotation and use_lateral:
    # Add both offsets together!
    y_arc = y_lateral_arc + y_rotation_arc
    y_line = y_lateral_line + y_rotation_line
```

**Example:**
```
Command: linear.y = 0.02, angular.z = 0.015
Result: Robot moves left AND rotates simultaneously!
```

## How Combining Works

When both lateral and rotation are active:

```python
# Lateral component
if leg == "FR" or leg == "HR":  # Right legs
    y_lateral_arc = -y_lat_f
else:  # Left legs
    y_lateral_arc = y_lat_b

# Rotation component
if leg == "HR" or leg == "FL":
    y_rotation_arc = y_rot_f
else:
    y_rotation_arc = y_rot_b

# Combine by adding
y_arc = y_lateral_arc + y_rotation_arc
```

The offsets are **added together**, allowing simultaneous movement!

## Direction Flags

Each movement type has its own direction flag:

```python
# Lateral direction
y_lat_f, y_lat_b = (back_y_lateral, front_y_lateral) if self.backwards_y else (front_y_lateral, back_y_lateral)

# Rotation direction
y_rot_f, y_rot_b = (back_y_rotation, front_y_rotation) if self.backwards_rot else (front_y_rotation, back_y_rotation)
```

**No more confusion!**

## Example Scenarios

### Scenario 1: Move Left
```
Command: linear.y = 0.02

use_lateral = True
use_rotation = False

Result:
  - Uses y_lat_f, y_lat_b
  - Pure lateral movement
  - Robot moves left ✓
```

### Scenario 2: Rotate CW
```
Command: angular.z = 0.015

use_lateral = False
use_rotation = True

Result:
  - Uses y_rot_f, y_rot_b
  - Pure rotation
  - Robot rotates in place ✓
```

### Scenario 3: Move Left + Rotate CW
```
Command: linear.y = 0.02, angular.z = 0.015

use_lateral = True
use_rotation = True

Result:
  - Combines y_lateral + y_rotation
  - Simultaneous movement
  - Robot moves left while rotating ✓
```

### Scenario 4: Forward + Left + Rotate
```
Command: linear.x = 0.03, linear.y = 0.02, angular.z = 0.015

Result:
  - X: forward movement (off_x = 0.03)
  - Y: lateral + rotation combined
  - Robot moves diagonally forward-left while rotating ✓
```

## Benefits

✅ **No mixing** - lateral and rotation are independent  
✅ **Can combine** - both can work simultaneously  
✅ **Separate direction flags** - no confusion  
✅ **Proper addition** - offsets add together naturally  
✅ **All combinations work** - forward+left+rotate, etc.  

## Visual Representation

### Before (Broken)
```
Lateral Movement ──┐
                   ├─→ One or the other ❌
Rotation ──────────┘
```

### After (Fixed)
```
Lateral Movement ──┐
                   ├─→ Both can work together ✓
Rotation ──────────┘
```

## Testing

### Test 1: Pure Lateral
```bash
# Move left
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0, y: 0.02, z: 0}, angular: {x: 0, y: 0, z: 0}}"
# Expected: Moves left only ✓
```

### Test 2: Pure Rotation
```bash
# Rotate CCW
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0, y: 0, z: 0}, angular: {x: 0, y: 0, z: 0.015}}"
# Expected: Rotates only ✓
```

### Test 3: Combined
```bash
# Move left + rotate
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0, y: 0.02, z: 0}, angular: {x: 0, y: 0, z: 0.015}}"
# Expected: Moves left while rotating ✓
```

### Test 4: All Three
```bash
# Forward + left + rotate
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 0.03, y: 0.02, z: 0}, angular: {x: 0, y: 0, z: 0.015}}"
# Expected: Complex combined movement ✓
```

## Summary

✅ **Lateral uses `off_y`** - from `linear.y`  
✅ **Rotation uses `off_rot`** - from `angular.z`  
✅ **Never mixed** - separate splines  
✅ **Can combine** - offsets add together  
✅ **Independent directions** - separate backwards flags  

The robot now handles lateral and rotation movements correctly! 🎯
