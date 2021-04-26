import pyrealsense2.pyrealsense2 as rs

def getDepthPipeline(FRAME_WIDTH, FRAME_HEIGHT, id=None):
    pipeline = rs.pipeline()
    config = rs.config()
    try:
        if(id!=None):
            config.enable_device(id)
    except Exception as e:
        print(e)

    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared,1, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared,2, FRAME_WIDTH, FRAME_HEIGHT, rs.format.y8, 30)
    
    print("[INFO] Starting streaming...")
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()    
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, config, profile, align, depth_scale
    

