#define _USE_MATH_DEFINES

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <random>
#include <cmath>

int* generate_shuffle(int width, int height) {
    int* shuffle = new int[width * height * 2];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = ((y * width) + x) * 2;
            shuffle[i] = x;
            shuffle[i + 1] = y;
        }
    }

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> rand(0, width * height);

    for (int i = 0; i < width * height; i++) {
        int pos1 = rand(rng) * 2;
        int pos2 = rand(rng) * 2;

        int tx = shuffle[pos1];
        int ty = shuffle[pos1 + 1];
        shuffle[pos1] = shuffle[pos2];
        shuffle[pos1 + 1] = shuffle[pos2 + 1];

        shuffle[pos2] = tx;
        shuffle[pos2 + 1] = ty;
    }

    return shuffle;
}

float bilinear_gradient(cv::Mat& grad, cv::Vec<float, 2> x) {
    int width = grad.cols;
    int height = grad.rows;

    cv::Vec<float, 2> offset(0.5, 0.5);
    x = x - offset;

    float x11 = -255.0, x12 = -255.0, x21 = -255.0, x22 = -255.0;

    if (x[0] >= 0.0 && x[0] < width - 0.5 && x[1] >= 0.0 && x[1] < height - 0.5)
        x11 = (float)grad.at<double>((int)x[1], (int)x[0]);

    if (x[0] + 1.0 >= 0.0 && x[0] + 1.0 < width - 0.5 && x[1] >= 0.0 && x[1] < height - 0.5)
        x12 = (float)grad.at<double>((int)x[1], (int)(x[0] + 1.0));

    if (x[0] >= 0.0 && x[0] < width - 0.5 && x[1] + 1.0 >= 0.0 && x[1] + 1.0 < height - 0.5)
        x21 = (float)grad.at<double>((int)(x[1] + 1.0), (int)x[0]);

    if (x[0] + 1.0 >= 0.0 && x[0] + 1.0 < width - 0.5 && x[1] + 1.0 >= 0.0 && x[1] + 1.0 < height - 0.5)
        x22 = (float)grad.at<double>((int)(x[1] + 1.0), (int)(x[0] + 1.0));

    if (x11 + x12 + x21 + x22 == -255.0 * 4)
        return 0.0;

    // If this is the corner case, 3 of the 4 cells will be -255
    // Adding them together sums to a number below 0 and their maximum must be the value from
    // The cell thats within bounds
    if (x11 + x12 + x21 + x22 < 0) return std::max(x11, std::max(x12, std::max(x21, x22)));

    float lhs = -1.0;
    float rhs = -1.0;

    // LHS
    float dist_y = x[1] - (int)x[1];
    if (x11 != -255.0 && x21 != -255.0)
        lhs = (dist_y * x11) + ((1.0 - dist_y) * x21);

    // RHS
    if (x12 != -255.0 && x22 != -255.0)
        rhs = (dist_y * x12) + ((1.0 - dist_y) * x22);

    if (lhs == -1.0)
        return rhs;
    else if (rhs == -1.0)
        return lhs;
    else {
        float dist_x = x[0] - (int)x[0];
        return (dist_x * lhs) + ((1.0 - dist_x) * rhs);
    }
}

cv::Vec<float, 2> get_stroke_endpoint(cv::Vec<float, 2>& c, cv::Vec<float, 2>& dir, float length, cv::Mat& grad) {
    // We will assume that dir has already been normalized

    // Loop: stop if distance from center is greater than half the stroke length
    // Break: if grad is less then previous grad
    cv::Vec<float, 2> x(c[0], c[1]);
    cv::Vec<float, 2> diff = c - x;
    float lastSample = bilinear_gradient(grad, x);
    float newSample;
    while (cv::norm(diff) < length / 2.0) {
        cv::Vec<float, 2> temp = x + dir;
        newSample = bilinear_gradient(grad, temp);
        if (newSample < lastSample)
            return x;

        lastSample = newSample;
        x = temp;

        diff = c - x;
    }

    return x;
}

int main() {
    // Load the original image
    std::string path = "../flower.jpg";
    cv::Mat img = cv::imread(path);

    // Convert image to greyscale
    cv::Mat img_grey;
    cv::cvtColor(img, img_grey, cv::COLOR_BGR2GRAY);

    // Blur the image to remove edge noise
    cv::Mat img_blur;
    cv::GaussianBlur(img_grey, img_blur, cv::Size(15, 15), 0);

    cv::Mat sobelxy;
    cv::Sobel(img_blur, sobelxy, CV_64F, 0, 1, 5);

    int half_width = img.cols / 2;
    int half_height = img.rows / 2;

    cv::Mat img_half;
    cv::resize(img, img_half, cv::Size(half_width, half_height), 0.5, 0.5);

    // Initialize an random number gen for the stroke angles
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> random_angle(0, 72);

    // Stroke Properties
    float stroke_length = 5.0;
    int stroke_width = 5;

    // Generate a shuffle order for the stroke drawing order
    int* shuffle = generate_shuffle(half_width, half_height);
    cv::Mat output(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < half_height; i++) {
        for (int j = 0; j < half_width; j++) {
            // Get the shuffled position
            int ipos = ((i * half_width) + j) * 2;
            int x = shuffle[ipos];
            int y = shuffle[ipos + 1];

            unsigned char* img_p = img_half.ptr(y, x);  // Y first, X after

            // Set up the location for the center of the stroke
            float cx = x * 2;
            float cy = y * 2;
            cv::Vec<float, 2> c(cx, cy);

            // Choose a random angle for the stroke
            int angle = random_angle(rng);
            float angle_rad = (5.0 * M_PI * 2.0 * angle) / 360.0;
            float dx = cos(angle_rad);
            float dy = sin(angle_rad);
            cv::Vec<float, 2> dir(dx, dy);

            // Compute the two ends of the stroke
            cv::Vec<float, 2> x1 = get_stroke_endpoint(c, dir, stroke_length, sobelxy);

            dir = dir * -1.0;
            cv::Vec<float, 2> x2 = get_stroke_endpoint(c, dir, stroke_length, sobelxy);

            cv::Point start(x1[0], x1[1]);
            cv::Point end(x2[0], x2[1]);
            cv::Scalar colour(img_p[0], img_p[1], img_p[2]);

            // Draw the stroke
            cv::line(output, start, end, colour, stroke_width, cv::LINE_AA);
        }
    }

    cv::imwrite("output.png", output);

    delete[] shuffle;

    return 0;
}
