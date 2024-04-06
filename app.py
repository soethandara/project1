import os
import cv2
from PIL import Image
import csv

def check_image_size_and_similarity(correct_image_path, student_images_folder):
    # Open the correct image
    correct_image = Image.open(correct_image_path)
    correct_width, correct_height = correct_image.size

    # Get a list of image files in the folder
    student_image_files = [f for f in os.listdir(student_images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    # Initialize list to store similarity results
    similarity_results = []

    for student_image_name in student_image_files:
        # Construct full path to the student image
        student_image_path = os.path.join(student_images_folder, student_image_name)

        # Open the student image
        student_image = Image.open(student_image_path)
        student_width, student_height = student_image.size
        threshold = 95.00
        # Compare image sizes
        if (student_width, student_height) == (correct_width, correct_height):
            # Calculate similarity using SURF
            similarity_percentage = calculate_similarity_by_SURF(correct_image_path, student_image_path)
            similarity_results.append({'Image': student_image_name, 'Similarity (%)': similarity_percentage, 'Remark': similarity_percentage})
        else:
            print(f"{student_image_name}: Incorrect size")
            similarity_results.append({'Image': student_image_name, 'Similarity (%)': 0.0, 'Remark': 0.0})
    
    # Sort similarity results in descending order based on similarity percentage
    similarity_results.sort(key=lambda x: x['Similarity (%)'], reverse=True)

    # Output CSV file path
    output_csv_path = os.path.join(student_images_folder, 'similarity_results.csv')

    # Write sorted results to CSV file with two decimal places
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Image', 'Similarity (%)', "Remark"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in similarity_results:
            writer.writerow({'Image': result['Image'], 'Similarity (%)': f"{result['Similarity (%)']:.2f}", 'Remark': f"{result['Remark']:.2f}"})

    print("Similarity comparison completed. Results saved in:", output_csv_path)

def calculate_similarity_by_SURF(image1_path, image2_path):
    # Read images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    # Initialize SURF detector
    surf = cv2.xfeatures2d.SURF_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = surf.detectAndCompute(image1, None)
    keypoints2, descriptors2 = surf.detectAndCompute(image2, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Calculate similarity percentage
    similarity = len(good_matches) / max(len(keypoints1), len(keypoints2)) * 100

    return similarity

if __name__ == "__main__":
    correct_image_path = '/path/to/correct_image.png'
    student_images_folder = '/path/to/student_images/'
    print (student_images_folder)
    print (correct_image_path)
    #changes made by developer1
    check_image_size_and_similarity(correct_image_path, student_images_folder)
