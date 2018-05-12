from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np

class TLClassifier(object):
    def __init__(self, model_path = 'model/model.pb', labels_path = 'model/'):
        # Load a (frozen) Tensorflow model into memory.
        config = tf.ConfigProto()
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=self.detection_graph, config=config)

        # simple class mapping based on models/label_map.pbtxt
        self.label_map = [None, 'red', 'yellow', 'green', 'unknown']

        self.tensor_dict = {
            "image_tensor": self.detection_graph.get_tensor_by_name('image_tensor:0'),
            "detection_boxes": self.detection_graph.get_tensor_by_name('detection_boxes:0'),
            "detection_scores": self.detection_graph.get_tensor_by_name('detection_scores:0'),
            "detection_classes": self.detection_graph.get_tensor_by_name('detection_classes:0'),
            "num_detections": self.detection_graph.get_tensor_by_name('num_detections:0')
        }
        print("Loaded graph")
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Actual detection.
        with self.detection_graph.as_default():
            image_np_expanded = np.expand_dims(image, axis=0)
            output_dict = self.sess.run(self.tensor_dict, feed_dict={self.tensor_dict['image_tensor']: image_np_expanded})

            boxes = np.squeeze(output_dict['detection_boxes'])
            scores = np.squeeze(output_dict['detection_scores'])
            classes = np.squeeze(output_dict['detection_classes']).astype(np.int32)
            thresh = .50
            for i in range(boxes.shape[0]):
                if scores is None or scores[i] > thresh:
                    class_name = self.label_map[classes[i]]
                    self.current_light = TrafficLight.UNKNOWN

                    if class_name == 'red':
                        self.current_light = TrafficLight.RED
                    elif class_name == 'green':
                        self.current_light = TrafficLight.GREEN
                    elif class_name == 'yellow':
                        self.current_light = TrafficLight.YELLOW

                    return self.current_light

        return TrafficLight.UNKNOWN
