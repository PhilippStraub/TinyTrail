# C++ Utilities

## Status
**Coming Soon** - C++ utilities planned for future development.

## Planned Components

### camera_handler.hpp/cpp
- OpenCV-based camera capture
- Frame preprocessing
- Hardware acceleration support

### preprocessors.hpp/cpp
- Image resizing and normalization
- MobileNetV2 preprocessing
- Batch processing support

### data_logger.hpp/cpp
- Efficient binary logging
- CSV export functionality
- Drift event recording

## Usage (Planned)
```cpp
#include "utils/camera_handler.hpp"
#include "utils/preprocessors.hpp"

CameraHandler camera(640, 480);
auto frame = camera.capture();
auto processed = preprocessMobileNetV2(frame);
```