import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2


def create_pipeline(id=None, FRAME_WIDTH=1280, FRAME_HEIGHT=720):
    pipeline = rs.pipeline()
    config = rs.config()
    if id is not None:
        config.enable_device(id)

    print("[INFO] Enabling stream...")
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared,1, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared,2, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)

    # profile
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # allign
    align_to = rs.stream.color
    align = rs.align(align_to)
    print("[INFO] Camera ready.")

    return pipeline, config, depth_scale, align


def create_frame(pipeline=None, align=None, clipping_distance=None, collision_percent=None):
    # super frame
    _output = dict({
        "depth_image": None,
        "color_image": None,
        "ir_image_1": None,
        "ir_image_2": None,
        "bg_removed": None,
        "collision": False
    })
    try:
        frames = pipeline.wait_for_frames()
        # frames
        ir_frame_1 = frames.get_infrared_frame(1)
        ir_frame_2 = frames.get_infrared_frame(2)

        
        # universal color image
        if align is not None:
            aligned_frames = align.process(frames)# aligned_depth_frame is a 640x480 depth image
            depth_frame = aligned_frames.get_depth_frame() 
            color_frame = aligned_frames.get_color_frame()


            _output["color_image"] = np.asanyarray(color_frame.get_data())

            _output["depth_image"] = np.asanyarray(depth_frame.get_data())
            grey_color = 0
            depth_image_3d = np.dstack((_output["depth_image"],_output["depth_image"],_output["depth_image"])) #depth image is 1 channel, color is 3 channels
            
            scaled_size = (color_frame.width, color_frame.height)
            # expand image dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            image_expanded = np.expand_dims(_output["color_image"], axis=0)

            _output["bg_removed"] = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, _output["color_image"])

            _output["bg_removed"][_output["bg_removed"] != [0,0,0]] = 255 

            white_count = np.count_nonzero(_output["bg_removed"] == 255)
            black_count = np.count_nonzero(_output["bg_removed"] == 0)
            pixels = _output["bg_removed"].shape[0]*_output["bg_removed"].shape[1]*_output["bg_removed"].shape[2]
            if(white_count > pixels*collision_percent):
                _output["collision"] = True
            else:
                _output["collision"] = False
        else:
            depth_frame = frames.get_depth_frame() 
            color_frame = frames.get_color_frame()
            _output["color_image"] = np.asanyarray(color_frame.get_data())
            _output["depth_image"] = np.asanyarray(frames.get_data())
        

        _output["ir_image_1"] = cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(ir_frame_1.get_data()), alpha=1), cv2.COLOR_BGRA2GRAY) 
        _output["ir_image_2"] = cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(ir_frame_2.get_data()), alpha=1), cv2.COLOR_BGRA2GRAY) 

        _output["depth_image"] = cv2.applyColorMap(cv2.convertScaleAbs(_output["depth_image"], alpha=0.02), cv2.COLORMAP_JET)
    except Exception as e:
        print(e)
    return _output
    



    



