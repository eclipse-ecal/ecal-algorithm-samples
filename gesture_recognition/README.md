# Gesture Recognition Sample

This is a python sample algorithm, that subscribes to webcam images, detects hands in that images and publishes their gesture and detected points. They can then be visualized using Foxglove Studio.

![image](resources/hand_with_connections.png)
![image](resources/hand_without_connections.png)

pb2-files are created with lib grpcio. In CMD: `python -m grpc_tools.protoc -I. --python_out=. RecognizedGesture.proto`
