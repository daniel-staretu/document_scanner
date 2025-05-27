#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat openImage(string path)
{
    Mat output_image = imread(path, IMREAD_UNCHANGED);

    if (output_image.empty()) {
        cout << "Could not open or find the image\n";
        return Mat();
    }

    return output_image;
}

Mat closingMorphological(Mat input_image)
{
    Mat output_image;
    int morph_size = 2;

    Mat element = getStructuringElement(
        MORPH_RECT,
        Size(1 * morph_size + 1, 2 * morph_size + 1),
        Point(morph_size, morph_size));

    morphologyEx(input_image, output_image, MORPH_CLOSE, element,
        Point(-1, -1), 2);

    return output_image;
}

Mat toGrayscale(Mat input_image) 
{
    Mat output_image;
    if (input_image.channels() == 3) {
        cvtColor(input_image, output_image, COLOR_BGR2GRAY);
    }
    else {
        output_image = input_image.clone();
    }

    return output_image;
}

Mat thresholdingOperation(Mat input_image)
{
    Mat output_image;

    double thresh = 180;
    double maxval = 255;
    int type = THRESH_BINARY;

    threshold(input_image, output_image, thresh, maxval, type);

    return output_image;
}

Mat edgeDetection(Mat input_image)
{
    Mat output_image;

    double threshold1 = 50;
    double threshold2 = 150;

    Canny(input_image, output_image, threshold1, threshold2);

    return output_image;
}

Mat cornerDetection(Mat input_image, vector<Point>& sorted_corners)
{
    Mat output_image;
    vector<vector<Point>> contours;

    findContours(input_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double max_area = 0;
    size_t max_index = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_index = i;
        }
    }

    //Douglas-Peucker algorithm
    vector<Point> approx;
    double epsilon = 0.02 * arcLength(contours[max_index], true);
    approxPolyDP(contours[max_index], approx, epsilon, true);

    while (approx.size() > 4) {
        epsilon *= 1.1;
        approxPolyDP(contours[max_index], approx, epsilon, true);
    }

    if (approx.size() > 4) {
        vector<Point> best_quad;
        double max_quad_area = 0;
        for (size_t i = 0; i < approx.size(); i++) {
            for (size_t j = i + 1; j < approx.size(); j++) {
                for (size_t k = j + 1; k < approx.size(); k++) {
                    for (size_t l = k + 1; l < approx.size(); l++) {
                        vector<Point> quad = { approx[i], approx[j], approx[k], approx[l] };
                        double area = contourArea(quad);
                        if (area > max_quad_area) {
                            max_quad_area = area;
                            best_quad = quad;
                        }
                    }
                }
            }
        }
        approx = best_quad;
    }

    sorted_corners = vector<Point>(4, Point(0, 0));
    if (approx.size() == 4) {
        Point center(0, 0);
        for (const auto& pt : approx) center += pt;
        center.x /= 4;
        center.y /= 4;

        for (const auto& pt : approx) {
            if (pt.x < center.x && pt.y < center.y) sorted_corners[0] = pt; // top-left
            else if (pt.x > center.x && pt.y < center.y) sorted_corners[1] = pt; // top-right
            else if (pt.x > center.x && pt.y > center.y) sorted_corners[2] = pt; // bottom-right
            else sorted_corners[3] = pt; // bottom-left
        }

        for (const auto& pt : sorted_corners) {
            if (pt == Point(0, 0)) {
                sort(approx.begin(), approx.end(), [](const Point& a, const Point& b) { return a.y < b.y; });
                sort(approx.begin(), approx.begin() + 2, [](const Point& a, const Point& b) { return a.x < b.x; });
                sort(approx.begin() + 2, approx.end(), [](const Point& a, const Point& b) { return a.x > b.x; });
                sorted_corners[0] = approx[0];
                sorted_corners[1] = approx[1];
                sorted_corners[2] = approx[3];
                sorted_corners[3] = approx[2];
                break;
            }
        }
    }
    else {
        sorted_corners = approx;
    }

    if (input_image.channels() == 1) {
        cvtColor(input_image, output_image, COLOR_GRAY2BGR);
    }
    else {
        output_image = input_image.clone();
    }

    for (size_t i = 0; i < sorted_corners.size(); i++) {
        circle(output_image, sorted_corners[i], 10, Scalar(0, 0, 255), FILLED);
        putText(output_image, to_string(i), sorted_corners[i], FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        line(output_image, sorted_corners[i], sorted_corners[(i + 1) % sorted_corners.size()], Scalar(0, 255, 0), 2);
    }

    return output_image;
}

Mat perspectiveTransform(Mat input_image, const vector<Point>& corners)
{
    Mat output_image;
    vector<Point2f> src_pts, dst_pts;

    double width_top = norm(corners[1] - corners[0]);
    double width_bottom = norm(corners[2] - corners[3]);
    double max_width = max(width_top, width_bottom);

    double height_left = norm(corners[3] - corners[0]);
    double height_right = norm(corners[2] - corners[1]);
    double max_height = max(height_left, height_right);

    for (const auto& pt : corners) {
        src_pts.push_back(Point2f(pt));
    }

    dst_pts = {
        Point2f(0, 0),
        Point2f(max_width - 1, 0),
        Point2f(max_width - 1, max_height - 1),
        Point2f(0, max_height - 1)
    };

    Mat transform_matrix = getPerspectiveTransform(src_pts, dst_pts);

    warpPerspective(input_image, output_image, transform_matrix,
        Size((int)max_width, (int)max_height), INTER_LINEAR, BORDER_CONSTANT);

    return output_image;
}

int main(int argc, char** argv)
{
    string image_path = "C:/coding/document_scanner/images/img01.png";
    Mat original_image = openImage(image_path);
    imshow("Step 0: Source Image", original_image);

    if (original_image.empty()) {
        cerr << "Error: Could not load image from path: " << image_path << endl;
        return -1;
    }

    Mat closing_morph_image = closingMorphological(original_image);
    imshow("Step 1: After Closing Morphological Operation", closing_morph_image);

    Mat grayscale_image = toGrayscale(closing_morph_image);
    imshow("Step 2: After Grayscale Conversion", grayscale_image);

    Mat thresholded_image = thresholdingOperation(grayscale_image);
    imshow("Step 3: After Thresholding Operation", thresholded_image);

    Mat edge_image = edgeDetection(thresholded_image);
    imshow("Step 4: After Edge Detection", edge_image);

    vector<Point> detected_corners;
    Mat corners_image = cornerDetection(edge_image, detected_corners);
    imshow("Step 5: After Corner Detection", corners_image);

    Mat transformed_image = perspectiveTransform(original_image, detected_corners);
    imshow("Step 6: Perspective Transformed Image", transformed_image);

    waitKey();
    return 0;
}
