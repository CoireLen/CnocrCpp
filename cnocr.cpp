#include "cnocr.h"
#include <algorithm>
#include <locale>
cnocr::cnocr(/* args */)
{
    locale lc("zh_CN.UTF-8");
    locale::global(lc);
    //载入中文字符
    wchar_t buffer[100];
    wifstream ctc_char(ctc_path);
    while (!ctc_char.eof() )
    {
        ctc_char.getline(buffer,100);
        ctc_data.push_back(buffer[0]);
    }
}
nc::NdArray<uint8_t> cnocr::toNdArray(cv::Mat imgdata){
    unsigned int imgwight=imgdata.size[0];
    unsigned int imghight=imgdata.size[1];
    auto img= nc::zeros<unsigned char>(imgwight,imghight);
    auto pimg=img.data();
    auto pimgdata=imgdata.data;
    unsigned int i_max=imgwight*imghight;
    for (int i=0;i<i_max;i++){
        *(pimg+i)=*(pimgdata+i);
    }
    return img;
}
nc::NdArray<uint8_t> cnocr::read_img(string path){
    auto imgdata=cv::imread(path,cv::IMREAD_GRAYSCALE);
    auto img=toNdArray(imgdata);
    cout <<"Imgndarray:"<<img.shape()<<endl;
    cout <<"Img 17_27:"<<int(img(17,27))<<endl;
    return img;
}
cv::Mat cnocr::toMat(nc::NdArray<uint8_t> img){
    auto imgcol=img.column(0).shape().size();
    auto imgrow=img.row(0).shape().size();
    using namespace cv;
    Mat imgdata(imgcol,imgrow,CV_8UC1);
    unsigned int i_max=imgcol*imgrow;
    auto pimg=imgdata.data;
    auto pimgdata=img.data();
    for (int i=0;i<i_max;i++){
        *(pimg+i)=*(pimgdata+i);
    }
    return imgdata;
}
cv::Mat cnocr::toMat(nc::NdArray<float> img){
    auto imgcol=img.column(0).shape().size();
    auto imgrow=img.row(0).shape().size();
    using namespace cv;
    Mat imgdata(imgcol,imgrow,CV_8UC1);
    unsigned int i_max=imgcol*imgrow;
    auto pimg=imgdata.data;
    auto pimgdata=img.data();
    for (int i=0;i<i_max;i++){
        *(pimg+i)=(int8_t)*(pimgdata+i)*255.0;
    }
    return imgdata;
}
void cnocr::show_img(nc::NdArray<uint8_t> img){
    auto imgdata=toMat(img);
    cv::imshow("img",imgdata);
    cv::waitKey();
}
vector<wstring> cnocr::ocr(string path){
    auto img =read_img(path);
    vector<wstring> res;
    auto imgcol=img.column(0).shape().size();
    auto imgrow=img.row(0).shape().size();
    if (min(imgrow,imgcol) < 2){
        return res;
    }
    cout<<"img(col,row):"<<imgcol <<","<<imgrow<<endl;
    if (nc::mean(img)[0] < 145) // 把黑底白字的图片对调为白底黑字
    {
        //img = 255 - img;
        auto a=nc::empty<unsigned char>(imgcol, imgrow);
        for (int i=0;i<imgcol*imgrow;i++)
        {
            *(a.data()+i)=(uchar)255;
        }
        //nc::NdArray<nc::uint8>(imgcol,imgrow) i=255;
        img =nc::add(a,-img);
    }
    cout<<"Mean:"<<nc::mean(img)[0]<<endl;
    auto imgs=line_split(img);
    res=ocr_for_single_lines(imgs);
    return res;
}
vector<wstring> cnocr::ocr(cv::Mat inimg){
    auto img=toNdArray(inimg);
    vector<wstring> res;
    auto imgcol=img.column(0).shape().size();
    auto imgrow=img.row(0).shape().size();
    if (min(imgrow,imgcol) < 2){
        return res;
    }
    cout<<"img(col,row):"<<imgcol <<","<<imgrow<<endl;
    if (nc::mean(img)[0] < 145) // 把黑底白字的图片对调为白底黑字
    {
        //img = 255 - img;
        auto a=nc::empty<unsigned char>(imgcol, imgrow);
        for (int i=0;i<imgcol*imgrow;i++)
        {
            *(a.data()+i)=(uchar)255;
        }
        //nc::NdArray<nc::uint8>(imgcol,imgrow) i=255;
        img =nc::add(a,-img);
    }
    cout<<"Mean:"<<nc::mean(img)[0]<<endl;
    auto imgs=line_split(img);
    res=ocr_for_single_lines(imgs);
    return res;
}
vector<nc::NdArray<uint8_t>> cnocr::line_split(nc::NdArray<uint8_t> img){
    vector<nc::NdArray<uint8_t>> list;
    auto imgcol=img.column(0).shape().size();
    auto imgrow=img.row(0).shape().size();
    auto bij=img.row(0);
    for (int i=0;i<imgrow;i++){
        *(bij.data()+i)=255;
    }
    int lineforchar=0;
    int lineforcharstart=0;
    for (int i=0;i<imgcol;i++){
        auto res=nc::add(bij,-img.row(i));
        if (res.max()(0,0)>100){
            if (lineforchar+1==i){
                lineforchar++;
            }
            else{
                if (lineforchar-lineforcharstart>7){
                    int start=lineforcharstart;
                    int end=lineforchar;
                    if (start>0){
                        start-=1;
                    }
                    if (end<imgcol-1){
                        end+=1;
                    }
                    list.push_back(img(nc::Slice(start, end), nc::Slice(0, imgrow)));
                }
                lineforcharstart=i;
                lineforchar=i;
            }
        }
    }
    if (lineforchar-lineforcharstart>5){
        int start=lineforcharstart;
        int end=lineforchar;
        if (start>0){
            start-=1;
        }
        if (end<imgcol-1){
            end+=1;
        }
        list.push_back(img(nc::Slice(start, end), nc::Slice(0, imgrow)));
    }
    return list;
}
vector<wstring> cnocr::ocr_for_single_lines(vector<nc::NdArray<uint8_t>> imgs){
    vector<wstring>res;
    if (imgs.size()==0){
        return res;
    }
    cout<<"have lines x for ocr:"<<imgs.size()<<endl;
    //res = self.rec_model.recognize(img_list, batch_size=batch_size)
    for (auto img:imgs){
        auto imgcol=img.column(0).shape().size();
        auto imgrow=img.row(0).shape().size();
        cout <<"imgSize:"<<imgcol<<","<<imgrow<<endl;
        //等比例放图像至高度32
        auto imgmat=toMat(img);
        float ratio=imgcol/32.0;
        cv::Size sz((int)(imgrow/ratio),32);
        cv::Mat imgresize(sz,CV_8UC1);
        cv::resize(imgmat,imgresize,sz);
        //调用模型
        long long input_height=imgresize.size[0];
        long long input_width=imgresize.size[1];
        //run_ort_trt(input_width,input_height*input_width,imgresize.data);
        vector<void *>ret_data=modle.run(input_width,input_height*input_width,imgresize.data);
        //img.transpose(2,0,1); 将数据转换
        auto ncdata=nc::empty<double>(*(int64_t*)ret_data[0],6674);
        for (size_t i=0;i<*(int64_t*)ret_data[0]*6674;i++){
                *(ncdata.data()+i)=*((float*)ret_data[1]+i);
        }
        //probs = F.softmax(logits.permute(0, 2, 1), dim=1)
        auto probs=nc::special::softmax(ncdata,nc::Axis::COL);
        //best_path = torch.argmax(probs, dim=1)  # [N, T]
        auto best_path=nc::argmax(probs,nc::Axis::COL);
        //best_path=nc::add(mask,-best_path);
        //length_mask = gen_length_mask(input_lengths, probs.shape).to(device=probs.device)
        //argmax 最大的值
        res.push_back(ctc_best(best_path));
        
    }
    return res;
}
wstring cnocr::ctc_best(nc::NdArray<uint32_t> data){
    wstring res;
    for (auto i:data){
        if (i<6673){
            res.push_back(ctc_data[i]);
        }
    }
    return res;
}
cnocr::~cnocr()
{
}
