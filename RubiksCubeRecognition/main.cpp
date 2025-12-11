#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

using namespace std;
using namespace cv;

/*************************************************************
 * 数据结构
 *************************************************************/
struct ColorRange {
    Scalar minVal;     // LAB 最小阈值
    Scalar maxVal;     // LAB 最大阈值
    string name;       // 颜色名称
    Scalar drawColor;  // 绘制颜色
    char code;         // 颜色代码（用于输出）
};

struct ColorBlock {
    Point2f center;    // 色块中心点
    string colorName;  // 颜色名称
    Scalar colorValue; // 颜色值
    Rect boundingBox;  // 边界框
    int row;           // 3x3网格中的行索引
    int col;           // 3x3网格中的列索引
    double area;       // 面积
};

/*************************************************************
 * 图像加载类
 *************************************************************/
class ImageLoader {
public:
    // 加载图像
    Mat loadImage(const string& filename) {
        Mat img = imread(filename);
        if (img.empty()) {
            cout << "无法加载图像：" << filename << endl;
        }
        else {
            cout << "成功加载图像：" << filename << endl;
        }
        return img;
    }
};

/*************************************************************
 * 色块检测与分析类
 *************************************************************/
class CubeFaceAnalyzer {
private:
    vector<ColorRange> colorTable; // 六种魔方颜色阈值
    map<string, char> colorCodes;  // 颜色名称到代码的映射

public:
    CubeFaceAnalyzer() {
        // 初始化颜色阈值表，包含六个颜色的LAB范围
        colorTable = {
            {{  0,146, 92}, { 94,187,155}, "Red",    Scalar(0,0,255), 'R'},
            {{139, 80,146}, {255,111,255}, "Yellow", Scalar(0,255,255), 'Y'},
            {{ 82, 42,  0}, {177,101,169}, "Green",  Scalar(0,255,0), 'G'},
            {{  0,  0,  0}, {255,255, 94}, "Blue",   Scalar(255,0,0), 'B'},
            {{160,127, 90}, {226,177,110}, "White",  Scalar(255,255,255), 'W'},
            {{ 87,158,106}, {163,255,172}, "Pink",   Scalar(203,192,255), 'P'}
        };

        // 初始化颜色代码映射
        for (auto& c : colorTable) {
            colorCodes[c.name] = c.code;
        }
    }

    /*********************************************************
     * 绘制虚线轮廓
     *********************************************************/
    void drawDashedContour(Mat& img, const vector<Point>& contour, Scalar color) {
        int segments = 8;      // 增加分段数，让虚线更密集
        int thickness = 5;     // 线条粗度

        for (int i = 0; i < contour.size(); i++) {
            Point p1 = contour[i];
            Point p2 = contour[(i + 1) % contour.size()];

            for (int k = 0; k < segments; k++) {
                float t1 = k / (float)segments;
                float t2 = (k + 0.5f) / (float)segments;  // 虚线的一半长度

                Point s = p1 + (p2 - p1) * t1;
                Point e = p1 + (p2 - p1) * t2;

                line(img, s, e, color, thickness, LINE_AA); // 使用抗锯齿
            }
        }
    }

    /*********************************************************
     * 比较函数：用于色块排序（先按行，再按列）
     *********************************************************/
    static bool compareColorBlocks(const ColorBlock& a, const ColorBlock& b) {
        if (a.row != b.row) {
            return a.row < b.row;
        }
        return a.col < b.col;
    }

    /*********************************************************
     * 检测并分析所有颜色色块
     *********************************************************/
    vector<ColorBlock> analyzeCubeFace(Mat& img, Mat& outputImg, bool draw = true) {
        vector<ColorBlock> allBlocks;
        Mat imgLab;
        cvtColor(img, imgLab, COLOR_BGR2Lab); // 转LAB颜色空间

        for (auto& c : colorTable) {
            Mat mask;
            inRange(imgLab, c.minVal, c.maxVal, mask); // 阈值分割

            // 形态学处理去噪
            morphologyEx(mask, mask, MORPH_OPEN, Mat(), Point(-1, -1), 2);

            // 提取轮廓
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            for (int i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area < 15000 || area > 150000) continue; // 过滤掉小面积噪声轮廓、阴影轮廓

                // 轮廓逼近
                float peri = arcLength(contours[i], true);
                vector<Point> approx;
                approxPolyDP(contours[i], approx, 0.002 * peri, true);

                // 绘制虚线轮廓
                if (draw) {
                    drawDashedContour(outputImg, approx, c.drawColor);
                }

                // 计算中心点
                Moments m = moments(contours[i]);
                Point2f center(m.m10 / m.m00, m.m01 / m.m00);

                // 创建色块信息
                ColorBlock block;
                block.center = center;
                block.colorName = c.name;
                block.colorValue = c.drawColor;
                block.boundingBox = boundingRect(approx);
                block.area = area;

                allBlocks.push_back(block);

                // 标注颜色名称
                if (draw) {
                    Rect boundRect = block.boundingBox;

                    // 在文字下加黑色背景条（增强对比）
                    rectangle(outputImg, Point(boundRect.x - 2, boundRect.y - 25),
                        Point(boundRect.x + 80, boundRect.y), Scalar(0, 0, 0), FILLED);

                    // 白色文字，字号更大
                    putText(outputImg, c.name, Point(boundRect.x, boundRect.y - 5),
                        FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
                }
            }
        }

        // 如果找到了9个色块，将它们分配到3x3网格
        if (allBlocks.size() == 9) {
            assignToGrid(allBlocks);
        }
        else {
            cout << "警告：检测到 " << allBlocks.size() << " 个色块，期望9个" << endl;
        }

        return allBlocks;
    }

    /*********************************************************
     * 将色块分配到3x3网格
     *********************************************************/
    void assignToGrid(vector<ColorBlock>& blocks) {
        // 按照Y坐标（行）排序
        vector<Point2f> centers;
        for (auto& block : blocks) {
            centers.push_back(block.center);
        }

        // 找到Y坐标的分割点（将9个点分成3行）
        vector<float> yCoords;
        for (auto& center : centers) {
            yCoords.push_back(center.y);
        }
        sort(yCoords.begin(), yCoords.end());

        // 确定行阈值
        float rowThreshold1 = yCoords[2] + (yCoords[3] - yCoords[2]) / 2;
        float rowThreshold2 = yCoords[5] + (yCoords[6] - yCoords[5]) / 2;

        // 为每个色块分配行索引
        for (auto& block : blocks) {
            if (block.center.y < rowThreshold1) {
                block.row = 0;
            }
            else if (block.center.y < rowThreshold2) {
                block.row = 1;
            }
            else {
                block.row = 2;
            }
        }

        // 对每一行按照X坐标（列）排序
        for (int r = 0; r < 3; r++) {
            vector<ColorBlock*> rowBlocks;
            for (auto& block : blocks) {
                if (block.row == r) {
                    rowBlocks.push_back(&block);
                }
            }

            // 按X坐标排序
            sort(rowBlocks.begin(), rowBlocks.end(),
                [](ColorBlock* a, ColorBlock* b) {
                    return a->center.x < b->center.x;
                });

            // 分配列索引
            for (int c = 0; c < rowBlocks.size(); c++) {
                rowBlocks[c]->col = c;
            }
        }

        // 按网格位置排序
        sort(blocks.begin(), blocks.end(), compareColorBlocks);
    }

    /*********************************************************
     * 创建颜色矩阵（3x3）并打印
     *********************************************************/
    vector<vector<char>> createColorMatrix(const vector<ColorBlock>& blocks) {
        vector<vector<char>> colorMatrix(3, vector<char>(3, ' '));

        for (const auto& block : blocks) {
            if (block.row >= 0 && block.row < 3 && block.col >= 0 && block.col < 3) {
                colorMatrix[block.row][block.col] = colorCodes[block.colorName];
            }
        }

        return colorMatrix;
    }

    /*********************************************************
     * 获取颜色映射表
     *********************************************************/
    map<string, Scalar> getColorMap() const {
        map<string, Scalar> colorMap;
        for (const auto& c : colorTable) {
            colorMap[c.name] = c.drawColor;
        }
        return colorMap;
    }

    /*********************************************************
     * 获取颜色代码映射表
     *********************************************************/
    map<char, Scalar> getColorCodeMap() const {
        map<char, Scalar> colorCodeMap;
        for (const auto& c : colorTable) {
            colorCodeMap[c.code] = c.drawColor;
        }
        return colorCodeMap;
    }
};

/*************************************************************
 * 色块标准化和魔方展开图绘制类
 *************************************************************/
class CubeVisualizer {
private:
    int blockSize = 60; // 每个色块的大小（像素）

public:
    /*********************************************************
     * 绘制单个标准化的魔方面
     *********************************************************/
    Mat drawStandardFace(const vector<vector<char>>& colorMatrix,
        const map<char, Scalar>& colorCodeMap,
        const string& faceName = "") {
        int margin = 10;
        int labelHeight = faceName.empty() ? 0 : 25;

        // 创建图像
        Mat faceImg(blockSize * 3 + margin * 2 + labelHeight,
            blockSize * 3 + margin * 2,
            CV_8UC3,
            Scalar(240, 240, 240));  // 浅灰色背景

        // 绘制每个色块
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                char colorCode = colorMatrix[row][col];

                // 获取颜色
                Scalar blockColor = Scalar(128, 128, 128); // 默认灰色
                if (colorCodeMap.find(colorCode) != colorCodeMap.end()) {
                    blockColor = colorCodeMap.at(colorCode);
                }

                // 计算位置
                int x = margin + col * blockSize;
                int y = margin + row * blockSize;

                // 绘制色块
                rectangle(faceImg,
                    Rect(x, y, blockSize, blockSize),
                    blockColor,
                    FILLED);

                // 绘制边框
                rectangle(faceImg,
                    Rect(x, y, blockSize, blockSize),
                    Scalar(50, 50, 50),  // 深灰色边框
                    2);

                // 在中心绘制颜色代码
                string codeStr(1, colorCode);
                putText(faceImg,
                    codeStr,
                    Point(x + blockSize / 3, y + 2 * blockSize / 3),
                    FONT_HERSHEY_SIMPLEX,
                    0.7,
                    Scalar(0, 0, 0),  // 黑色文字
                    2);
            }
        }

        // 添加面名称标签（如果有）
        if (!faceName.empty()) {
            putText(faceImg,
                faceName,
                Point(margin, faceImg.rows - margin / 2),
                FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar(0, 0, 0),  // 黑色文字
                2);
        }

        return faceImg;
    }

    /*********************************************************
     * 绘制魔方展开图（使用标准4x3网格布局）
     *********************************************************/
    Mat drawCubeNet(const vector<vector<vector<char>>>& allColorMatrices,
        const map<char, Scalar>& colorCodeMap) {
        int margin = 10;

        // 计算画布尺寸：4x3网格布局
        int cols = 4 * (blockSize * 3 + margin) + margin;
        int rows = 3 * (blockSize * 3 + margin) + margin;
        Mat cubeNet = Mat::zeros(rows, cols, CV_8UC3);
        cubeNet.setTo(Scalar(240, 240, 240)); // 浅灰色背景

        // 标准魔方展开图布局：
        // 位置: (列, 行) 每个单位是一个面的大小+边距
        vector<Point> facePositions = {
            Point(1, 0),  // Up: 第0行第1列
            Point(0, 1),  // Left: 第1行第0列
            Point(1, 1),  // Front: 第1行第1列
            Point(2, 1),  // Right: 第1行第2列
            Point(3, 1),  // Back: 第1行第3列
            Point(1, 2)   // Down: 第2行第1列
        };

        vector<string> faceLabels = { "Up", "Left", "Front", "Right", "Back", "Down" };

        for (int i = 0; i < allColorMatrices.size() && i < 6; i++) {
            Point pos = facePositions[i];
            int x = margin + pos.x * (blockSize * 3 + margin);
            int y = margin + pos.y * (blockSize * 3 + margin);

            // 绘制单个面
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 3; c++) {
                    char colorCode = allColorMatrices[i][r][c];
                    Scalar color = Scalar(128, 128, 128); // 默认灰色

                    if (colorCodeMap.find(colorCode) != colorCodeMap.end()) {
                        color = colorCodeMap.at(colorCode);
                    }

                    Rect blockRect(x + c * blockSize, y + r * blockSize, blockSize, blockSize);
                    rectangle(cubeNet, blockRect, color, FILLED);
                    rectangle(cubeNet, blockRect, Scalar(50, 50, 50), 2);

                    // 显示颜色代码
                    string codeStr(1, colorCode);
                    putText(cubeNet, codeStr,
                        Point(x + c * blockSize + blockSize / 4,
                            y + r * blockSize + 3 * blockSize / 4),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
                }
            }

            // 添加面标签
            putText(cubeNet, faceLabels[i], Point(x + 5, y - 5),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
        }

        return cubeNet;
    }

    /*********************************************************
     * 创建检测结果与标准化结果的对比图
     *********************************************************/
    Mat createComparisonImage(const Mat& detectionImg,
        const Mat& standardFace,
        const string& faceName) {
        // 调整图像尺寸以匹配
        Mat resizedDetection, resizedStandard;
        int targetHeight = 400;
        int targetWidth = 400;

        resize(detectionImg, resizedDetection, Size(targetWidth, targetHeight));
        resize(standardFace, resizedStandard, Size(targetWidth, targetHeight));

        // 创建组合图像
        Mat combined(targetHeight, targetWidth * 2, CV_8UC3, Scalar(240, 240, 240));

        // 将检测图像放在左侧
        Mat leftROI = combined(Rect(0, 0, targetWidth, targetHeight));
        resizedDetection.copyTo(leftROI);

        // 将标准化图像放在右侧
        Mat rightROI = combined(Rect(targetWidth, 0, targetWidth, targetHeight));
        resizedStandard.copyTo(rightROI);



        return combined;
    }
};

/*************************************************************
 * 主程序
 *************************************************************/
int main() {
    ImageLoader loader;
    CubeFaceAnalyzer analyzer;
    CubeVisualizer visualizer;

    // 获取颜色映射
    map<char, Scalar> colorCodeMap = analyzer.getColorCodeMap();

    // 六张魔方图像
    vector<string> filenames = {
        "data/cubeface1.jpg",
        "data/cubeface2.jpg",
        "data/cubeface3.jpg",
        "data/cubeface4.jpg",
        "data/cubeface5.jpg",
        "data/cubeface6.jpg"
    };

    vector<string> faceNames = { "Front", "Back", "Left", "Right", "Up", "Down" };
    vector<vector<vector<char>>> allColorMatrices; // 存储所有面的颜色矩阵
    vector<Mat> standardFaces; // 存储标准化后的面

    // 创建输出目录
    system("mkdir -p output");

    cout << "===== 魔方颜色检测程序 =====" << endl;
    cout << "注意：请确保图像文件位于 data/ 目录下" << endl << endl;

    // 处理每个面
    for (int i = 0; i < filenames.size(); i++) {
        cout << "\n============== 处理第 " << (i + 1) << " 张图 (" << faceNames[i] << ") ==============\n";

        // 1) 加载图像
        Mat img = loader.loadImage(filenames[i]);
        if (img.empty()) continue;

        // 2) 克隆原图，用于处理
        Mat processedImg = img.clone();

        // 3) 分析魔方面，检测色块
        vector<ColorBlock> blocks = analyzer.analyzeCubeFace(img, processedImg, true);

        cout << "检测到 " << blocks.size() << " 个色块" << endl;

        // 4) 创建颜色矩阵并打印
        vector<vector<char>> colorMatrix = analyzer.createColorMatrix(blocks);

        cout << "颜色矩阵 (" << faceNames[i] << "):" << endl;
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                cout << colorMatrix[row][col] << " ";
            }
            cout << endl;
        }

        // 5) 保存颜色矩阵
        allColorMatrices.push_back(colorMatrix);

        // 6) 创建标准化面
        Mat standardFace = visualizer.drawStandardFace(colorMatrix, colorCodeMap, faceNames[i]);
        standardFaces.push_back(standardFace);

        // 7) 创建检测结果与标准化结果的对比图
        Mat comparison = visualizer.createComparisonImage(processedImg, standardFace, faceNames[i]);

        // 8) 显示和保存结果
        string windowName = "Face " + to_string(i + 1) + " - " + faceNames[i];
        imshow(windowName, comparison);

        // 保存处理结果
        string processedPath = "output/processed_" + faceNames[i] + ".jpg";
        string standardPath = "output/standard_" + faceNames[i] + ".jpg";
        string comparisonPath = "output/comparison_" + faceNames[i] + ".jpg";

        imwrite(processedPath, processedImg);
        imwrite(standardPath, standardFace);
        imwrite(comparisonPath, comparison);

        cout << "处理结果已保存：" << endl;
        cout << "  - 检测图: " << processedPath << endl;
        cout << "  - 标准化图: " << standardPath << endl;
        cout << "  - 对比图: " << comparisonPath << endl;

        cout << "\n按任意键继续处理下一个面..." << endl;
        waitKey(0);
        destroyAllWindows();
    }

    // 9) 绘制完整的魔方展开图
    if (allColorMatrices.size() >= 1) {
        // 重新排序颜色矩阵以匹配展开图布局：Up, Left, Front, Right, Back, Down
        vector<vector<vector<char>>> reorderedMatrices;
        vector<string> reorderedNames = { "Up", "Left", "Front", "Right", "Back", "Down" };

        // 原始顺序：Front, Back, Left, Right, Up, Down
        // 新顺序：Up(4), Left(2), Front(0), Right(3), Back(1), Down(5)
        vector<int> order = { 4, 2, 0, 3, 1, 5 };

        for (int idx : order) {
            if (idx < allColorMatrices.size()) {
                reorderedMatrices.push_back(allColorMatrices[idx]);
            }
        }

        Mat cubeNet = visualizer.drawCubeNet(reorderedMatrices, colorCodeMap);
        imshow("魔方展开图", cubeNet);

        // 保存魔方展开图
        imwrite("output/cube_net.jpg", cubeNet);
        cout << "\n魔方展开图已保存到 output/cube_net.jpg" << endl;

        // 输出所有面的颜色代码
        cout << "\n============== 所有面的颜色代码 ==============\n";
        for (int i = 0; i < allColorMatrices.size(); i++) {
            cout << "\n面 " << faceNames[i] << ":" << endl;
            for (int row = 0; row < 3; row++) {
                for (int col = 0; col < 3; col++) {
                    cout << allColorMatrices[i][row][col] << " ";
                }
                cout << endl;
            }
        }

        // 输出颜色分组统计
        cout << "\n============== 颜色分组统计 ==============\n";
        map<char, int> colorCount;
        for (const auto& matrix : allColorMatrices) {
            for (const auto& row : matrix) {
                for (char color : row) {
                    if (color != ' ') {
                        colorCount[color]++;
                    }
                }
            }
        }

        cout << "颜色分布（每个颜色应有9个色块）：" << endl;
        for (const auto& pair : colorCount) {
            cout << "颜色 " << pair.first << ": " << pair.second << " 个色块" << endl;
        }

        waitKey(0);
        destroyAllWindows();
    }

    cout << "\n===== 程序执行完成 =====" << endl;
    cout << "所有结果已保存到 output/ 目录下" << endl;

    return 0;
}