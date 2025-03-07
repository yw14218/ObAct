import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import config
import asyncio
from asyncio import Future

class CartesianVisualServoer(Node):
    def __init__(self, use_depth=False):
        super().__init__('visual_servoer')
        
        self.bridge = CvBridge()
        self.use_depth = use_depth
        
        self.future = Future()
        
        self.rgb_sub = message_filters.Subscriber(self, Image, config.d405_rgb_topic_name)
        self.depth_sub = message_filters.Subscriber(self, Image, config.d405_depth_topic_name) if self.use_depth else None
        
        if self.use_depth:
            self.sync = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1)
            self.sync.registerCallback(self.synchronized_callback)
        else:
            self.rgb_sub.registerCallback(self.rgb_callback)

    def rgb_callback(self, rgb_msg):
        if not self.future.done():
            try:
                rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                self.future.set_result((rgb_image, None))
            except Exception as e:
                self.get_logger().error(f"RGB callback error: {e}")

    def synchronized_callback(self, rgb_msg, depth_msg):
        if not self.future.done():
            try:
                rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
                self.future.set_result((rgb_image, depth_image))
            except Exception as e:
                self.get_logger().error(f"Synchronization callback error: {e}")

    async def observe(self):
        self.future = Future()
        try:
            return await self.future
        except Exception:
            self.get_logger().warn("Timeout occurred while waiting for synchronized images.")
            return (None, None)

    async def observe_loop(self):
        while rclpy.ok():
            rgb, depth = await self.observe()
            if rgb is not None:
                self.get_logger().info("RGB Image received.")
            if depth is not None:
                self.get_logger().info("Depth Image received.")

async def main():
    rclpy.init()
    node = CartesianVisualServoer(use_depth=True)
    
    loop = asyncio.get_running_loop()
    loop.create_task(node.observe_loop())
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    asyncio.run(main())



