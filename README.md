# stereo_depth (ROS2 Jazzy)

Bu paket, yüklediğin orijinal stereo-vision projesini ROS2 Jazzy paketine port
eder. Çalıştığında:

1. `/camera0/image_raw` (sol) ve `/camera1/image_raw` (sağ) topiclerinden
   time-synchronised olarak görüntü alır.
2. `/camera0/camera_info` ve `/camera1/camera_info` topiclerinden kamera
   intrinsic'lerini (özellikle focal length **f**) okur.
3. **İlk frame'de** orijinal koddaki gibi
   `SIFT → BFMatcher → RANSAC (8-point) → cv2.stereoRectifyUncalibrated`
   adımlarını çalıştırarak rectification homografilerini (H1, H2) hesaplar
   ve cache'ler. (Her frame'de bunu tekrarlamak gerçek-zamanlı çalışmayı
   imkânsız hale getirir.)
4. Her frame'de:
   * Her iki görüntü H1, H2 ile warp edilir,
   * `cv2.StereoSGBM` ile disparity haritası üretilir,
   * `depth_mm = (baseline_mm × f_px) / disparity_px` ile derinlik hesaplanır.
5. Açılan OpenCV penceresinde fare ile bir alan seçildiğinde, alanın
   içindeki **geçerli derinliklerin medyanı** mm ve m cinsinden görüntünün
   üstüne yazılır.

> **Not (önemli):** Orijinal `stereo.py` içindeki naive block-matching
> döngüsü (çift `for` + numpy broadcast) çoğu çözünürlükte saniyede 1
> frame bile zor üretiyor, bu yüzden disparity için OpenCV'nin
> StereoSGBM'i kullanıldı. Rectification kısmında **algoritma birebir
> orijinal kod**: `normalize` + 8-point + RANSAC fonksiyonları
> `stereo_depth/utils/fundamental_matrix.py` içinde port edildi.

---

## Baseline

Konuşmada belirtildiği gibi varsayılan **baseline = 60 mm**. Launch
argümanı `baseline_mm` ile değiştirilebilir.

## Bağımlılıklar

```bash
sudo apt install ros-jazzy-cv-bridge ros-jazzy-message-filters \
                 ros-jazzy-sensor-msgs python3-opencv python3-numpy
```

> `cv2.SIFT_create` modern OpenCV'de standart — `opencv-contrib-python`
> gerekmez (ama varsa da sorun değil).

## Build

```bash
# workspace'inin src/ klasörüne kopyala
cd ~/ros2_ws/src
# (bu klasörü buraya yerleştir)

cd ~/ros2_ws
colcon build --packages-select stereo_depth
source install/setup.bash
```

## Çalıştırma

```bash
# launch dosyası ile (önerilen)
ros2 launch stereo_depth stereo_depth.launch.py

# veya tek node olarak
ros2 run stereo_depth stereo_node
```

### Topic / parametre override örnekleri

```bash
ros2 launch stereo_depth stereo_depth.launch.py \
    baseline_mm:=60.0 \
    left_image_topic:=/camera0/image_raw \
    right_image_topic:=/camera1/image_raw \
    left_info_topic:=/camera0/camera_info \
    right_info_topic:=/camera1/camera_info \
    sgbm_num_disparities:=128 \
    sgbm_block_size:=5
```

## Pencere kontrolleri

| İşlem | Kısayol |
|---|---|
| Dikdörtgen seç | **Sol tık + sürükle** |
| Daire seç | **Sağ tık + sürükle** |
| Seçimi temizle | **c** |
| Rectification'ı yeniden hesapla (sahne değiştiyse) | **k** |
| Çıkış | **q** |

İki pencere açılır:
* **Stereo Depth (left rectified)** — ana pencere, seçim ve mesafe burada.
* **Disparity (debug)** — renkli disparity haritası
  (launch arg `show_disparity_window:=false` ile kapatılabilir).

## Topic / parametre özeti

### Subscribed topics
| Topic | Tip | Açıklama |
|---|---|---|
| `/camera0/image_raw` | `sensor_msgs/Image` | Sol kamera |
| `/camera1/image_raw` | `sensor_msgs/Image` | Sağ kamera |
| `/camera0/camera_info` | `sensor_msgs/CameraInfo` | Sol intrinsic |
| `/camera1/camera_info` | `sensor_msgs/CameraInfo` | Sağ intrinsic |

### Parametreler
| Ad | Varsayılan | Açıklama |
|---|---|---|
| `baseline_mm` | `60.0` | Kameralar arası mesafe (mm) |
| `left_image_topic` | `/camera0/image_raw` | |
| `right_image_topic` | `/camera1/image_raw` | |
| `left_info_topic` | `/camera0/camera_info` | |
| `right_info_topic` | `/camera1/camera_info` | |
| `sync_slop` | `0.05` | ApproximateTime senkron toleransı (s) |
| `sgbm_num_disparities` | `128` | 16'nın katı olmalı |
| `sgbm_block_size` | `5` | Tek sayı, tipik 3..11 |
| `show_disparity_window` | `true` | Debug penceresi |

## Paket yapısı

```
stereo_depth/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/stereo_depth
├── launch/
│   └── stereo_depth.launch.py
└── stereo_depth/
    ├── __init__.py
    ├── stereo_node.py            # ROS2 node, mouse, görselleştirme
    ├── stereo_processor.py       # rectify + disparity + depth pipeline
    └── utils/
        ├── __init__.py
        ├── fundamental_matrix.py # orijinal: FundamentalEssentialMatrix.py
        └── misc_utils.py         # orijinal: MiscUtils.py
```

## Bilinen sınırlamalar / ipuçları

* **Uncalibrated rectification** sahnedeki feature dağılımına çok
  duyarlıdır. Düz duvar / texture'suz sahne ile feature bulamayabilir;
  bu durumda `k` ile farklı bir sahnede yeniden kalibre etmeyi dene.
* Gerçek bir stereo rig'in varsa, **çok daha sağlamı**: kameraları
  `camera_calibration` paketi ile bir kez kalibre edip
  `cv2.stereoRectify` + `initUndistortRectifyMap` kullanmak. Sen
  "kaynaktaki gibi uncalibrated" dediğin için bu paket onu yapıyor; ama
  ihtiyaç olursa bunu kalibre versiyona çevirmek bir günlük iş.
* `baseline_mm` ile `CameraInfo.k[0]` (fx) **aynı birimle birleşince**
  derinlik **mm** çıkar — sayısal sıhhat için fx'in pixel cinsinden
  doğru yayınlandığından emin ol.
* SGBM `numDisparities` arttıkça yakın objeleri daha iyi görür ama
  yavaşlar; `blockSize` arttıkça gürültüye dayanım artar, ince
  detaylar yok olur.

## Lisans
MIT (orijinal proje referans: https://github.com/sakshikakde/Depth-Using-Stereo)
# stereo_depth
