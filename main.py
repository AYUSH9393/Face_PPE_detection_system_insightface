"""
Face Recognition System - Main Entry Point

This is the main script to run the face recognition system.
It provides a command-line interface for various operations.
"""

import os
import sys
import argparse
import cv2

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from face_encoder import FaceEncoder
from face_recognizer import FaceRecognizer
from video_processor import VideoProcessor
from utils import load_image, draw_face_box, draw_landmarks, save_image


def build_database(known_faces_dir: str = "known_faces", database_path: str = "database/faces.pkl"):
    """
    Build face database from known faces directory
    """
    print("="*60)
    print("Building Face Database")
    print("="*60)
    
    encoder = FaceEncoder()
    database = encoder.build_database_from_directory(known_faces_dir, database_path)
    
    if len(database) == 0:
        print("\n⚠ Warning: No faces were added to the database!")
        print(f"Please add images to the '{known_faces_dir}' directory.")
        print("\nExpected structure:")
        print(f"{known_faces_dir}/")
        print("  ├── person1/")
        print("  │   ├── image1.jpg")
        print("  │   └── image2.jpg")
        print("  └── person2/")
        print("      └── image1.jpg")
    
    return database


def recognize_image(image_path: str, database_path: str = "database/faces.pkl", 
                   output_path: str = None, show: bool = True, threshold: float = 0.4):
    """
    Recognize faces in a single image
    """
    print("="*60)
    print("Face Recognition - Image")
    print("="*60)
    
    # Initialize recognizer
    recognizer = FaceRecognizer(database_path, similarity_threshold=threshold)
    
    # Load and process image
    print(f"\nProcessing image: {image_path}")
    image = load_image(image_path)
    results = recognizer.recognize_faces_in_image(image)
    
    # Draw results
    output_image = image.copy()
    
    print(f"\nDetected {len(results)} face(s):")
    for i, result in enumerate(results):
        print(f"\n  Face {i+1}:")
        print(f"    Name: {result['name']}")
        print(f"    Similarity: {result['similarity']:.3f}")
        print(f"    Detection score: {result['det_score']:.3f}")
        
        # Choose color
        color = (0, 255, 0) if result['is_recognized'] else (0, 0, 255)
        
        # Draw bounding box
        output_image = draw_face_box(
            output_image,
            result['bbox'],
            label=result['name'],
            confidence=result['similarity'],
            color=color
        )
        
        # Draw landmarks
        if result['landmarks'] is not None:
            output_image = draw_landmarks(output_image, result['landmarks'])
    
    # Save output
    if output_path:
        save_image(output_image, output_path)
        print(f"\n✓ Output saved to: {output_path}")
    
    # Display
    if show:
        cv2.imshow('Face Recognition Result', output_image)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return results


def recognize_video(video_path: str, database_path: str = "database/faces.pkl",
                   output_path: str = None, show: bool = True, threshold: float = 0.4,
                   use_gpu: bool = True, log_detections: bool = True):
    """
    Recognize faces in a video file
    """
    print("="*60)
    print("Face Recognition - Video")
    print("="*60)
    
    # Initialize recognizer and processor with GPU support
    recognizer = FaceRecognizer(database_path, similarity_threshold=threshold, use_gpu=use_gpu)
    processor = VideoProcessor(recognizer, show_landmarks=True)
    
    # Generate log file path if logging is enabled
    log_path = None
    if log_detections:
        import os
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        log_path = f"output/{video_name}_detections.txt"
    
    # Process video
    stats = processor.process_video_file(video_path, output_path, display=show, log_path=log_path)
    
    return stats


def recognize_webcam(database_path: str = "database/faces.pkl", 
                    output_path: str = None, threshold: float = 0.4, camera_id: int = 0):
    """
    Recognize faces from webcam feed
    """
    print("="*60)
    print("Face Recognition - Webcam")
    print("="*60)
    
    # Initialize recognizer and processor
    recognizer = FaceRecognizer(database_path, similarity_threshold=threshold)
    processor = VideoProcessor(recognizer, show_landmarks=True)
    
    # Process webcam
    processor.process_webcam(camera_id=camera_id, output_path=output_path)


def show_database_info(database_path: str = "database/faces.pkl"):
    """
    Display information about the face database
    """
    print("="*60)
    print("Face Database Information")
    print("="*60)
    
    from utils import load_database
    
    database = load_database(database_path)
    
    if len(database) == 0:
        print("\n⚠ Database is empty!")
        print("Run 'python main.py build' to create a database.")
        return
    
    print(f"\nDatabase path: {database_path}")
    print(f"Total identities: {len(database)}")
    
    total_embeddings = sum(data['num_images'] for data in database.values())
    print(f"Total embeddings: {total_embeddings}")
    
    print("\nIdentities:")
    for name, data in database.items():
        print(f"  - {name}: {data['num_images']} image(s)")


def main():
    """
    Main function with command-line interface
    """
    parser = argparse.ArgumentParser(
        description="Face Recognition System using InsightFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build database from known faces
  python main.py build
  
  # Recognize faces in an image
  python main.py image path/to/image.jpg
  
  # Recognize faces in a video
  python main.py video path/to/video.mp4 -o output/result.mp4
  
  # Start webcam recognition
  python main.py webcam
  
  # Show database information
  python main.py info
        """
    )
    
    parser.add_argument('mode', choices=['build', 'image', 'video', 'webcam', 'info'],
                       help='Operation mode')
    parser.add_argument('input', nargs='?', help='Input file path (for image/video mode)')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-d', '--database', default='database/faces.pkl',
                       help='Path to face database (default: database/faces.pkl)')
    parser.add_argument('-k', '--known-faces', default='known_faces',
                       help='Directory with known faces (default: known_faces)')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                       help='Similarity threshold (default: 0.4)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display output')
    parser.add_argument('-c', '--camera', type=int, default=0,
                       help='Camera ID for webcam mode (default: 0)')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU acceleration (default: True)')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage (overrides --gpu)')
    parser.add_argument('--no-log', action='store_true',
                       help='Disable detection logging for video mode')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'build':
            build_database(args.known_faces, args.database)
        
        elif args.mode == 'image':
            if not args.input:
                parser.error("Image path required for 'image' mode")
            
            output = args.output if args.output else f"output/result_{os.path.basename(args.input)}"
            recognize_image(args.input, args.database, output, not args.no_display, args.threshold)
        
        elif args.mode == 'video':
            if not args.input:
                parser.error("Video path required for 'video' mode")
            
            # Determine GPU usage
            use_gpu = not args.cpu  # Use GPU unless --cpu is specified
            
            output = args.output if args.output else "output/output_video.mp4"
            recognize_video(args.input, args.database, output, not args.no_display, 
                          args.threshold, use_gpu=use_gpu, log_detections=not args.no_log)
        
        elif args.mode == 'webcam':
            output = args.output if args.output else "output/webcam_recording.mp4"
            recognize_webcam(args.database, output, args.threshold, args.camera)
        
        elif args.mode == 'info':
            show_database_info(args.database)
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
