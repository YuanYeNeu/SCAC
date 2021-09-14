/*
 *This program is modified based on the Compact Watershed codes, by YUAN Ye, yuanye_neu@163.com.
 *If you use these codes, please cite the correspoding paper: Superpixels with Content-adaptive Criteria
 *
 *This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 *This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *We would like to thank Peer Neubert et al. for their work.
 *The original statement is shown as below.
 */
/*
 * Compact Watershed
 * Copyright (C) 2014  Peer Neubert, peer.neubert@etit.tu-chemnitz.de
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
//#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"
#include <string>
#include <vector>
#include <iostream>
//#include<cv.h>
//#include<highgui.h>
#include<iostream>
//#include "sys/time.h"
#include <time.h>
#include "mex_helper_SCAC.h"
#include "mex_SCAC.h"

using namespace std;
using namespace cv;

/*
 * call: B = mex_compact_watershed(uint8(I), n, compactness, single(seeds))
 *       B = mex_compact_watershed(uint8(I), n, compactness)
 *
 * I            ... input image
 * n            ... number of segments
 * compactness  ... compactness parameter, e.g. 1.0
 * seeds        ... matrix of initial seeds, each row is [i,j], single values (optional)
 *
 * B ... resulting boundary image
 *
 * compile with
 *  mex mex_compact_watershed.cpp  mex_helper.cpp ../compact_watershed.cpp $(pkg-config --cflags --libs opencv)
 */



void compact_watershed(Mat& img, Mat& B, Mat& seeds, int spn,int ItrSet,int SM, int lambda, int * labelnumber)
{   Mat markers = Mat::zeros(img.rows, img.cols, CV_32SC1);
    int labelIdx = 0;
    
        float d=sqrt(float(img.rows*img.cols)/float(spn));
        int nx=round(img.cols/d);
        int ny=round(img.rows/d);
       int disAver=ceil(d);
        
        
        
        for(int i=0;i<img.rows;i++)
        {
            for(int j=0;j < img.cols; j++)
            {
                int noj=ceil(j/d);
                if (noj>nx)
                    noj=nx;
                int noi=floor(i/d);
                if (noi>=ny)
                    noi=ny-1;
                
                markers.at<int>(i, j)=nx*noi+noj;
                
            }
        }
        
        
        
        labelIdx=nx*ny;
    
    cvWatershed( img, markers,labelIdx,spn,disAver,ItrSet,SM, lambda,labelnumber);
    
    
    
    // create boundary map
    B=markers;
    
    
    // extend boundary map to image borders
    
}

typedef struct CvWSNode
{
    struct CvWSNode* next;
    int mask_ofs;
    int img_ofs;
    float compVal;
}
CvWSNode;

typedef struct CvWSQueue
{
    CvWSNode* first;
    CvWSNode* last;
}
CvWSQueue;

static CvWSNode* icvAllocWSNodes(CvMemStorage* storage)
{
    CvWSNode* n = 0;
    
    int i, count = (storage->block_size - sizeof(CvMemBlock)) / sizeof(*n) - 1;
    
    n = (CvWSNode*)cvMemStorageAlloc(storage, count*sizeof(*n));
    for (i = 0; i < count - 1; i++)
        n[i].next = n + i + 1;
    n[count - 1].next = 0;
    
    return n;
}

void  cvWatershed(Mat& imgRGB, Mat& BMat,  int labelId,int spn, int disAver,int itrSet,int SM, int lambda, int* labelnumber)
{
    Mat imgMat;
    imgMat=imgRGB.clone();
    Mat imgBlur=imgRGB.clone();
    GaussianBlur(imgRGB, imgBlur, Size(3, 3), 0, 0);
    cvtColor(imgBlur, imgMat, COLOR_BGR2Lab);
    const int IN_QUEUE = -2;
    const int WSHED = -1;
    const int RELABELED=-4;
    const int NQ =  2048;
    int mmq=10;
    if (ceil(0.5*disAver)>10)
        mmq=ceil(0.5*disAver);
    const int MQ=mmq;
    int Improve=1;
    cv::Ptr<CvMemStorage> storage;
    Mat labelMat=Mat::zeros(labelId, 3, CV_32SC1);
    CvMat sstub;
    CvMat dstub;
    CvSize size;
    CvWSNode* free_node = 0, *node;
    CvWSQueue q[NQ+1];
    int active_queue;
    int i, j;
    double db, dg, dr;
    int* mask;
    int* iflabeled;
    uchar* img;
    uchar* oriimg;
    int* labelLab;
    int* numberLab;
    int mstep, istep,labstep;
    int subs_tab[2 * NQ + 1];
    
    Mat numberMat=Mat::zeros(labelId, 1, CV_32SC1);
    int imgrow=imgMat.rows;
    int imgcol=imgMat.cols;
    
    // MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
    // MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])
    
#define ws_push(idx,mofs,iofs,cV)  \
    {                               \
                                            if (!free_node)            \
                                            free_node = icvAllocWSNodes(storage); \
                                            node = free_node;           \
                                                    free_node = free_node->next; \
                                                    node->next = 0;             \
                                                            node->mask_ofs = mofs;      \
                                                            node->img_ofs = iofs;       \
                                                                    node->compVal = cV;    \
                                                                    if (q[idx].last)           \
                                                                            q[idx].last->next = node; \
                                                                    else                        \
                                                                            q[idx].first = node;    \
                                                                            q[idx].last = node;         \
    }
    
#define ws_pop(idx,mofs,iofs,cV)   \
    {                               \
                                            node = q[idx].first;        \
                                            q[idx].first = node->next;  \
                                                    if (!node->next)           \
                                                    q[idx].last = 0;        \
                                                    node->next = free_node;     \
                                                            free_node = node;           \
                                                            mofs = node->mask_ofs;      \
                                                                    iofs = node->img_ofs;       \
                                                                    cV = node->compVal;       \
    }
    
#define cc_diff(ptr0,lTemp,aTemp,bTemp,t)      \
    {                                   \
                                                db = double((ptr0)[0])-double(lTemp); \
                                                dg = double((ptr0)[1])-double(aTemp); \
                                                        dr =double((ptr0)[2])-double(bTemp); \
                                                        t = double(sqrt(0.1*db*db+1.45*dg*dg+1.45*dr*dr)); \
    }
    
    
    
    
    Mat iflabelMat=Mat::zeros(imgrow,imgcol,CV_32SC1);
    CvMat *src = cvCreateMat(imgrow,imgcol,CV_8UC3);
    CvMat *dst = cvCreateMat(imgrow,imgcol,CV_32SC1);
    CvMat *labellabptr = cvCreateMat(labelId,3,CV_32SC1);
    CvMat *numberlabptr = cvCreateMat(labelId,1,CV_32SC1);
    CvMat *iflabel = cvCreateMat(imgrow,imgcol,CV_32SC1);
    
    
    CvMat temp = imgMat; 
    cvCopy(& temp, src);
    CvMat temp0 = BMat;
    cvCopy(& temp0, dst);
    temp0=iflabelMat;
    cvCopy(& temp0, iflabel);
    CvMat temp000=labelMat;
    cvCopy(& temp000, labellabptr);
    CvMat temp0000=numberMat;
    cvCopy(& temp0000, numberlabptr);
    if (CV_MAT_TYPE(src->type) != CV_8UC3)
        CV_Error(CV_StsUnsupportedFormat, "Only 8-bit, 3-channel input images are supported");
    
    if (CV_MAT_TYPE(dst->type) != CV_32SC1)
        CV_Error(CV_StsUnsupportedFormat,
                "Only 32-bit, 1-channel output images are supported");
    
    if (!CV_ARE_SIZES_EQ(src, dst))
        CV_Error(CV_StsUnmatchedSizes, "The input and output images must have the same size");
    
    size = cvGetMatSize(src);
    storage = cvCreateMemStorage();
    
    istep = src->step;
    img = src->data.ptr;
    mstep = dst->step / sizeof(mask[0]);
    mask = dst->data.i;
    
    iflabeled=iflabel->data.i;
    labstep=labellabptr->step/sizeof(labelLab[0]);
    labelLab=labellabptr->data.i;
    numberLab=numberlabptr->data.i;
    
    memset(q, 0, (NQ+1)*sizeof(q[0]));
    
    for (i = 0; i < NQ; i++)
        subs_tab[i] = 0;
    for (i = NQ; i <= 2 * NQ; i++)
        subs_tab[i] = i - NQ;
    
    
    for(i=0;i<imgrow;i++)
    {
        iflabeled[i*imgcol]=WSHED;
        iflabeled[i*imgcol+imgcol-1]=WSHED;
        mask[i*imgcol]=WSHED;
        mask[i*imgcol+imgcol-1]=WSHED;
    }
    for(j=0;j < imgcol; j++)
    {
        iflabeled[j]=WSHED;
        iflabeled[imgrow*imgcol-imgcol+j]=WSHED;
        mask[j]=WSHED;
        mask[imgrow*imgcol-imgcol+j]=WSHED;
    }
    
    for(i=1;i<size.height-1;i++)
    {
        for(j=1;j < size.width-1; j++)
        {
            //iflabeled[i*size.width+j]=0;
            if(mask[i*size.width+j]>0)
            {
                int tempmask=mask[i*size.width+j];
                labelLab[tempmask*labstep-3]+=img[3*i*size.width+3*j];
                labelLab[tempmask*labstep-2]+=img[3*i*size.width+3*j+1];
                labelLab[tempmask*labstep-1]+=img[3*i*size.width+3*j+2];
                numberLab[tempmask-1]++;
            }
        }
    }
    int  label1=labelId;
    
    int gg=6;
   
    {
        int labelpri;
        CvMat *cvGrey = cvCreateMat(imgrow,imgcol,CV_8UC1);
        uchar *ptrGrey;
        Mat imageGray;
        Mat imgColor=imgRGB.clone();
        GaussianBlur( imgBlur, imgColor, Size(3,3), 0, 0, BORDER_DEFAULT );
        cvtColor(imgColor,imageGray,CV_BGR2GRAY);
        CvMat ImgGrey = imageGray;
        cvCopy(& ImgGrey, cvGrey);
        ptrGrey = cvGrey->data.ptr;
        
        cv::Mat imgGradient=Mat::zeros(imgrow, imgcol, CV_8UC1);
        //Mat imgGradient=imgGrey;
        Mat kernelWLD = (Mat_<char>(3, 3) <<1, 1, 1, 1, -8, 1, 1, 1, 1);
        filter2D(imageGray, imgGradient, -1, kernelWLD);
        CvMat *cvimgGradient = cvCreateMat(imgrow, imgcol, CV_8UC1);
        uchar *ptrimgGradient;
        CvMat imageGradient=imgGradient;
        cvCopy(&imageGradient, cvimgGradient);
        ptrimgGradient=cvimgGradient->data.ptr;
        for(i=0;i<imgrow;i++)
        {
            for(j=0;j < imgcol; j++)
            {
                double intensity;
                if  (ptrGrey[i*imgcol+j]==0)
                    intensity=0.00000001;
                else
                    intensity=ptrGrey[i*imgcol+j];
                double textureFilter=double(ptrimgGradient[i*imgcol+j])/intensity;

                ptrimgGradient[i*imgcol+j]=atan(textureFilter)*100;
            }
        }
        
     
        if(Improve==1)
        {
            cv::Mat GreyMeanMat0=Mat::zeros(labelId, 1, CV_32SC1);
            cv::Mat GreyNumMat0=Mat::zeros(labelId, 1, CV_32SC1);
            CvMat *cvGreyMean0 = cvCreateMat(labelId,1,CV_32SC1);
            int *ptrGreyMean0;
            CvMat *cvGreyNum0 = cvCreateMat(labelId,1,CV_32SC1);
            int *ptrGreyNum0;
            CvMat GreyMean0=GreyMeanMat0;
            cvCopy(& GreyMean0, cvGreyMean0);
            CvMat GreyNum0=GreyNumMat0;
            cvCopy(& GreyNum0, cvGreyNum0);
            ptrGreyMean0=cvGreyMean0->data.i;
            ptrGreyNum0=cvGreyNum0->data.i;
            
            cv::Mat GradientMat=Mat::zeros(imgrow, imgcol, CV_32SC1);
            CvMat *cvGradient = cvCreateMat(imgrow,imgcol,CV_8UC1);
            uchar *ptrGradient;
            Mat grad;
            Mat grad_x, grad_y;
            Mat abs_grad_x, abs_grad_y;
            int scale = 1;
            int delta = 0;
            int ddepth = CV_16S;
            
            Sobel( imageGray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_x, abs_grad_x );
            Sobel( imageGray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
            convertScaleAbs( grad_y, abs_grad_y );
            addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
            CvMat tempgrad =grad;
            cvCopy(& tempgrad, cvGradient);
            ptrGradient=cvGradient->data.ptr;
            
            //BMat = Mat(cvGradient, true);
            
            cv::Mat PositionMat=Mat::zeros(labelId, 2, CV_32SC1);
            CvMat *cvPosition = cvCreateMat(labelId,2,CV_32SC1);
            int *ptrPosition;
            CvMat Position=PositionMat;
            cvCopy(& Position, cvPosition);
            ptrPosition=cvPosition->data.i;
            
            
            //累计各SP的颜色值、中心、和pixel个数
            for(i=1;i<imgrow-1;i++)
            {
                for(j=1;j < imgcol-1; j++)
                {
                    int maskvalue=mask[i*imgcol+j];
                    int* m;
                    m=mask+i*imgcol+j;
                    ptrGreyMean0[maskvalue-1]+=ptrGrey[i*imgcol+j];
                    ptrGreyNum0[maskvalue-1]++;
                    ptrPosition[2*maskvalue-2]+=i;
                    ptrPosition[2*maskvalue-1]+=j;
                }
            }
            
            for(i=1;i<imgrow-1;i++)
            {
                for(j=1;j < imgcol-1 ; j++)
                {
                    iflabeled[i*imgcol+j]=0;
                }
            }
            //global marching
            int nC=9999999999;
            int itr=0;
            for (; nC>label1&&itr<itrSet;)
                // for (int itr=1; itr<itrSet+1;itr++)
            {
                nC=0;
                itr++;
                
                
                int tagL=0-itr;
                
                for(i=1;i<imgrow-1;i++)
                {
                    for(j=1;j < imgcol-1; j++)
                    {
                        int maskvalue=mask[i*imgcol+j];
                        int ma=i*imgcol+j;
                        int flagEdge=0;
                        if (mask[ma+1]>maskvalue)
                            flagEdge=1;
                        else if (mask[ma-1]>maskvalue)
                            flagEdge=1;
                        else if (mask[ma+imgcol]>maskvalue)
                            flagEdge=1;
                        else if (mask[ma-imgcol]>maskvalue)
                            flagEdge=1;
                        
                        if (flagEdge==1)
                        {
                            
                            ws_push(0,ma, maskvalue, i);
                            iflabeled[ma]=tagL;
                        }
                    }
                }
                active_queue=0;
                for (;;)
                {
                    int i_ori,j_ori;
                    int mofs, valuePri;
                    int qqq;
                    if (q[active_queue].first == 0)
                    {
                        for (qqq = active_queue + 1; qqq < MQ+1; qqq++)
                            if (q[qqq].first)
                                break;
                        if (qqq > MQ)
                            break;
                        active_queue = qqq;
                    }
                    ws_pop(active_queue, mofs, valuePri, i_ori);
                    j_ori=mofs-imgcol*i_ori;
                    uchar* ptr;
                    ptr = img + 3*mofs;
                    iflabeled[mofs]=itr;
                    int valueNei;
                    if (mask[mofs+1]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs+mstep]!=valuePri&&(mask[mofs-mstep]!=valuePri||mask[mofs-mstep+1]!=valuePri||mask[mofs-mstep-1]!=valuePri))
                    {
                        iflabeled[mofs]=0;
                    }
                    else if (mask[mofs+1]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs-mstep]!=valuePri&&(mask[mofs+mstep]!=valuePri||mask[mofs+mstep+1]!=valuePri||mask[mofs+mstep-1]!=valuePri))
                    {
                        iflabeled[mofs]=0;
                    }
                    else if (mask[mofs+mstep]==valuePri&&mask[mofs-mstep]==valuePri&&mask[mofs-1]!=valuePri&&(mask[mofs+mstep+1]!=valuePri||mask[mofs-mstep+1]!=valuePri||mask[mofs+1]!=valuePri))
                    {
                        iflabeled[mofs]=0;
                    }
                    else if (mask[mofs+mstep]==valuePri&&mask[mofs-mstep]==valuePri&&mask[mofs+1]!=valuePri&&(mask[mofs+mstep-1]!=valuePri||mask[mofs-mstep-1]!=valuePri||mask[mofs-1]!=valuePri))
                    {
                        iflabeled[mofs]=0;
                    }
                    else if (mask[mofs+mstep]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs+mstep-1]!=valuePri)
                    {
                        iflabeled[mofs]=0;
                    }
                    else if (mask[mofs+mstep]==valuePri&&mask[mofs+1]==valuePri&&mask[mofs+mstep+1]!=valuePri)
                    {
                        iflabeled[mofs]=0;
                    }
                    else if (mask[mofs-mstep]==valuePri&&mask[mofs+1]==valuePri&&mask[mofs-mstep+1]!=valuePri)
                    {
                        iflabeled[mofs]=0;
                    }
                    else if (mask[mofs-mstep]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs-mstep-1]!=valuePri)
                    {
                        iflabeled[mofs]=0;
                    }
                    else
                    {
                        double spaceC=double(lambda);
                        double tempC=100.000;
                        int labT=valuePri;
                        double minT=100000000000.0000;
                        int numpri=ptrGreyNum0[valuePri-1]-1;
                        int flagN=0;
                        if(numpri==0)
                        {
                            numpri=1;
                            flagN=1;
                        }
                        double deltapri=0.000;
                        int lTemp;
                        int aTemp;
                        int bTemp;
                        if (flagN==1)
                            deltapri=0;
                        else
                        {
                            
                            aTemp=(labelLab[valuePri*labstep-2]-ptr[1])/numpri;
                            bTemp=(labelLab[valuePri*labstep-1]-ptr[2])/numpri;
                            lTemp=(labelLab[valuePri*labstep-3]-ptr[0])/numpri;
                            
                            cc_diff(ptr,lTemp,aTemp,bTemp,deltapri);
                        }
                        double i_pri=double(ptrPosition[2*valuePri-2]-i_ori)/double(numpri);
                        double j_pri=double(ptrPosition[2*valuePri-1]-j_ori)/double(numpri);
                        double distancePri=sqrt(double(abs(i_ori-i_pri)*abs(i_ori-i_pri)+abs(j_ori-j_pri)*abs(j_ori-j_pri)));
                        double ePri=tempC*deltapri+spaceC*distancePri;
                        
                        valueNei = mask[mofs-1];
                        
                        if (valueNei> 0&&valueNei!=valuePri&&flagN==0)
                        {
                            double i_new=double(ptrPosition[2*valueNei-2])/double(ptrGreyNum0[valueNei-1]);
                            double j_new=double(ptrPosition[2*valueNei-1])/double(ptrGreyNum0[valueNei-1]);
                            aTemp=labelLab[valueNei*labstep-2]/ptrGreyNum0[valueNei-1];
                            bTemp=labelLab[valueNei*labstep-1]/ptrGreyNum0[valueNei-1];
                            lTemp=labelLab[valueNei*labstep-3]/ptrGreyNum0[valueNei-1];
                            double deltanei=0.000;
                            cc_diff(ptr,lTemp,aTemp,bTemp,deltanei);
                            double distanceNei=sqrt(double(abs(i_ori-i_new)*abs(i_ori-i_new)+abs(j_ori-j_new)*abs(j_ori-j_new)));
                            
                            double eNei=spaceC*distanceNei+tempC*deltanei;
                            if (eNei<ePri)
                            {
                                labT=valueNei;
                                minT=eNei;
                            }
                        }
                        valueNei = mask[mofs+1];
                        if (valueNei> 0&&valueNei!=valuePri&&flagN==0)
                        {
                            double i_new=double(ptrPosition[2*valueNei-2])/double(ptrGreyNum0[valueNei-1]);
                            double j_new=double(ptrPosition[2*valueNei-1])/double(ptrGreyNum0[valueNei-1]);
                            aTemp=labelLab[valueNei*labstep-2]/ptrGreyNum0[valueNei-1];
                            bTemp=labelLab[valueNei*labstep-1]/ptrGreyNum0[valueNei-1];
                            lTemp=labelLab[valueNei*labstep-3]/ptrGreyNum0[valueNei-1];
                            double deltanei=0.000;
                            cc_diff(ptr,lTemp,aTemp,bTemp,deltanei);
                            
                            double distanceNei=sqrt(double(abs(i_ori-i_new)*abs(i_ori-i_new)+abs(j_ori-j_new)*abs(j_ori-j_new)));
                            double eNei=spaceC*distanceNei+tempC*deltanei;
                            if (eNei<minT)
                            {
                                
                                if (eNei<ePri)
                                {
                                    labT=valueNei;
                                    minT=eNei;
                                    
                                    //deltaEdge=deltaE;
                                }
                            }
                        }
                        valueNei = mask[mofs+mstep];
                        if (valueNei> 0&&valueNei!=valuePri&&flagN==0)
                        {
                            double i_new=double(ptrPosition[2*valueNei-2])/double(ptrGreyNum0[valueNei-1]);
                            double j_new=double(ptrPosition[2*valueNei-1])/double(ptrGreyNum0[valueNei-1]);
                            aTemp=labelLab[valueNei*labstep-2]/ptrGreyNum0[valueNei-1];
                            bTemp=labelLab[valueNei*labstep-1]/ptrGreyNum0[valueNei-1];
                            lTemp=labelLab[valueNei*labstep-3]/ptrGreyNum0[valueNei-1];
                            double deltanei=0.000;
                            cc_diff(ptr,lTemp,aTemp,bTemp,deltanei);
                            
                            double distanceNei=sqrt(double(abs(i_ori-i_new)*abs(i_ori-i_new)+abs(j_ori-j_new)*abs(j_ori-j_new)));
                            double eNei=spaceC*distanceNei+tempC*deltanei;
                            
                            if (eNei<minT)
                            {
                                
                                if (eNei<ePri)
                                {
                                    labT=valueNei;
                                    minT=eNei;
                                }
                            }
                        }
                        valueNei = mask[mofs-mstep];
                        if (valueNei> 0&&valueNei!=valuePri&&flagN==0)
                        {
                            double i_new=double(ptrPosition[2*valueNei-2])/double(ptrGreyNum0[valueNei-1]);
                            double j_new=double(ptrPosition[2*valueNei-1])/double(ptrGreyNum0[valueNei-1]);
                            aTemp=labelLab[valueNei*labstep-2]/ptrGreyNum0[valueNei-1];
                            bTemp=labelLab[valueNei*labstep-1]/ptrGreyNum0[valueNei-1];
                            lTemp=labelLab[valueNei*labstep-3]/ptrGreyNum0[valueNei-1];
                            double deltanei=0.000;
                            cc_diff(ptr,lTemp,aTemp,bTemp,deltanei);
                            
                            double distanceNei=sqrt(double(abs(i_ori-i_new)*abs(i_ori-i_new)+abs(j_ori-j_new)*abs(j_ori-j_new)));
                            double eNei=spaceC*distanceNei+tempC*deltanei;
                            if (eNei<minT)
                            {
                                
                                if (eNei<ePri)
                                {
                                    labT=valueNei;
                                    minT=eNei;
                                }
                            }
                        }
                        mask[mofs]=labT;
                        
                        if (valuePri!=mask[mofs])
                        {
                            nC++;
                            int labT=mask[mofs];
                            ptrGreyMean0[labT-1]+=ptrGrey[mofs];
                            ptrGreyMean0[valuePri-1]-=ptrGrey[mofs];
                            ptrGreyNum0[valuePri-1]--;
                            ptrGreyNum0[labT-1]++;
                            ptrPosition[2*valuePri-2]=ptrPosition[2*valuePri-2]-i_ori;
                            ptrPosition[2*valuePri-1]=ptrPosition[2*valuePri-1]-j_ori;
                            ptrPosition[2*labT-2]=ptrPosition[2*labT-2]+i_ori;
                            ptrPosition[2*labT-1]=ptrPosition[2*labT-1]+j_ori;
                            labelLab[labT*labstep-1]+=ptr[2];
                            labelLab[labT*labstep-2]+=ptr[1];
                            labelLab[labT*labstep-3]+=ptr[0];
                            labelLab[valuePri*labstep-1]-=ptr[2];
                            labelLab[valuePri*labstep-2]-=ptr[1];
                            labelLab[valuePri*labstep-3]-=ptr[0];
                            valuePri=labT;
                            
                        }
                    }
                    //将新的边界点存入queue
                    if (active_queue<MQ-1)
                    {
                        valueNei = mask[mofs-1];
                        if(valueNei!=valuePri&&iflabeled[mofs-1]!=tagL&&iflabeled[mofs-1]!=itr&&valueNei>0)
                        {
                            ws_push(active_queue+1,mofs-1, valueNei, i_ori);
                            iflabeled[mofs-1]=tagL;
                        }
                        valueNei = mask[mofs+1];
                        if(valueNei!=valuePri&&iflabeled[mofs+1]!=tagL&&iflabeled[mofs+1]!=itr&&valueNei>0)
                        {
                            ws_push(active_queue+1,mofs+1, valueNei, i_ori);
                            iflabeled[mofs+1]=tagL;
                        }
                        valueNei = mask[mofs-mstep];
                        if(valueNei!=valuePri&&iflabeled[mofs-mstep]!=tagL&&iflabeled[mofs-mstep]!=itr&&valueNei>0)
                        {
                            ws_push(active_queue+1,mofs-mstep, valueNei, i_ori-1);
                            iflabeled[mofs-mstep]=tagL;
                        }
                        valueNei = mask[mofs+mstep];
                        if(valueNei!=valuePri&&iflabeled[mofs+mstep]!=tagL&&iflabeled[mofs+mstep]!=itr&&valueNei>0)
                        {
                            ws_push(active_queue+1,mofs+mstep, valueNei, i_ori+1);
                            iflabeled[mofs+mstep]=tagL;
                        }
                    }
                    
                    
                }
               
                
              
                {
                    for(i=1;i<imgrow-1;i++)
                    {
                        for(j=1;j < imgcol-1; j++)
                        {
                            int mofs=i*imgcol+j;
                            int valuePri=mask[mofs];
                            int i_ori=i;
                            int j_ori=j;
                            
                            for(;;)
                            {
                                if (mask[mofs+1]==valuePri&&mask[mofs-1]!=valuePri&&mask[mofs-mstep]!=valuePri&&mask[mofs+mstep]!=valuePri)
                                {
                                    
                                    if (ptrGradient[mofs]<gg||(mask[mofs+1+mstep]!=valuePri&&mask[mofs+1-mstep]!=valuePri&&(abs(ptrGrey[mofs+1]-ptrGrey[mofs+1+mstep])<gg||abs(ptrGrey[mofs+1]-ptrGrey[mofs+1-mstep])<gg)))
                                    {
                                        int del=10000;
                                        int labT;
                                        int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                        if (mask[mofs-1]>0&&del0<del)
                                        {
                                            del=del0;
                                            labT=mask[mofs-1];
                                        }
                                        del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                        if (del0<del&&mask[mofs-mstep]>0)
                                        {
                                            del=del0;
                                            labT=mask[mofs-mstep];
                                        }
                                        del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                        if (del0<del&&mask[mofs+mstep]>0)
                                        {
                                            //del=del0;
                                            labT=mask[mofs+mstep];
                                        }
                                        mask[mofs]=labT;
                                        
                                        
                                        ptrGreyMean0[labT-1]+=ptrGrey[mofs];
                                        ptrGreyMean0[valuePri-1]-=ptrGrey[mofs];
                                        ptrGreyNum0[valuePri-1]--;
                                        ptrGreyNum0[labT-1]++;
                                        ptrPosition[2*valuePri-2]=ptrPosition[2*valuePri-2]-i_ori;
                                        ptrPosition[2*valuePri-1]=ptrPosition[2*valuePri-1]-j_ori;
                                        ptrPosition[2*labT-2]=ptrPosition[2*labT-2]+i_ori;
                                        ptrPosition[2*labT-1]=ptrPosition[2*labT-1]+j_ori;
                                        uchar* ptr;
                                        ptr= img + 3*i_ori*imgcol+3*j_ori;
                                        labelLab[labT*labstep-3]+=ptr[0];
                                        labelLab[labT*labstep-2]+=ptr[1];
                                        labelLab[labT*labstep-1]+=ptr[2];
                                        labelLab[valuePri*labstep-3]-=ptr[0];
                                        labelLab[valuePri*labstep-2]-=ptr[1];
                                        labelLab[valuePri*labstep-1]-=ptr[2];
                                        j_ori+=1;
                                        mofs=mofs+1;
                                    }
                                    else
                                        break;
                                    
                                    
                                }
                                else if (mask[mofs-1]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-mstep]!=valuePri&&mask[mofs+mstep]!=valuePri)
                                {
                                    if (ptrGradient[mofs]<gg||(mask[mofs-1+mstep]!=valuePri&&mask[mofs-1-mstep]!=valuePri&&(abs(ptrGrey[mofs-1]-ptrGrey[mofs-1+mstep])<gg||abs(ptrGrey[mofs-1]-ptrGrey[mofs-1-mstep])<gg)))
                                    {
                                        int del=10000;
                                        int labT;
                                        int del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                        if (mask[mofs+1]>0&&del0<del)
                                        {
                                            del=del0;
                                            labT=mask[mofs+1];
                                        }
                                        del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                        if (del0<del&&mask[mofs-mstep]>0)
                                        {
                                            del=del0;
                                            labT=mask[mofs-mstep];
                                        }
                                        del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                        if (del0<del&&mask[mofs+mstep]>0)
                                        {
                                            //del=del0;
                                            labT=mask[mofs+mstep];
                                        }
                                        mask[mofs]=labT;
                                        
                                        ptrGreyMean0[labT-1]+=ptrGrey[mofs];
                                        ptrGreyMean0[valuePri-1]-=ptrGrey[mofs];
                                        ptrGreyNum0[valuePri-1]--;
                                        ptrGreyNum0[labT-1]++;
                                        ptrPosition[2*valuePri-2]=ptrPosition[2*valuePri-2]-i_ori;
                                        ptrPosition[2*valuePri-1]=ptrPosition[2*valuePri-1]-j_ori;
                                        ptrPosition[2*labT-2]=ptrPosition[2*labT-2]+i_ori;
                                        ptrPosition[2*labT-1]=ptrPosition[2*labT-1]+j_ori;
                                        uchar* ptr;
                                        ptr= img + 3*i_ori*imgcol+3*j_ori;
                                        labelLab[labT*labstep-3]+=ptr[0];
                                        labelLab[labT*labstep-2]+=ptr[1];
                                        labelLab[labT*labstep-1]+=ptr[2];
                                        labelLab[valuePri*labstep-3]-=ptr[0];
                                        labelLab[valuePri*labstep-2]-=ptr[1];
                                        labelLab[valuePri*labstep-1]-=ptr[2];
                                        j_ori-=1;
                                        mofs=mofs-1;
                                    }
                                    else
                                        break;
                                }
                                else if (mask[mofs-mstep]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-1]!=valuePri&&mask[mofs+mstep]!=valuePri)
                                {
                                    if (ptrGradient[mofs]<gg||(mask[mofs+1-mstep]!=valuePri&&mask[mofs-1-mstep]!=valuePri&&(abs(ptrGrey[mofs-mstep]-ptrGrey[mofs+1-mstep])<gg||abs(ptrGrey[mofs-mstep]-ptrGrey[mofs-1-mstep])<gg)))
                                    {
                                        int del=10000;
                                        int labT;
                                        int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                        if (mask[mofs-1]>0&&del0<del)
                                        {
                                            del=del0;
                                            labT=mask[mofs-1];
                                        }
                                        del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                        if (del0<del&&mask[mofs+1]>0)
                                        {
                                            del=del0;
                                            labT=mask[mofs+1];
                                        }
                                        del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                        if (del0<del&&mask[mofs+mstep]>0)
                                        {
                                            //del=del0;
                                            labT=mask[mofs+mstep];
                                        }
                                        mask[mofs]=labT;
                                        
                                        ptrGreyMean0[labT-1]+=ptrGrey[mofs];
                                        ptrGreyMean0[valuePri-1]-=ptrGrey[mofs];
                                        ptrGreyNum0[valuePri-1]--;
                                        ptrGreyNum0[labT-1]++;
                                        ptrPosition[2*valuePri-2]=ptrPosition[2*valuePri-2]-i_ori;
                                        ptrPosition[2*valuePri-1]=ptrPosition[2*valuePri-1]-j_ori;
                                        ptrPosition[2*labT-2]=ptrPosition[2*labT-2]+i_ori;
                                        ptrPosition[2*labT-1]=ptrPosition[2*labT-1]+j_ori;
                                        uchar* ptr;
                                        ptr= img + 3*i_ori*imgcol+3*j_ori;
                                        labelLab[labT*labstep-3]+=ptr[0];
                                        labelLab[labT*labstep-2]+=ptr[1];
                                        labelLab[labT*labstep-1]+=ptr[2];
                                        labelLab[valuePri*labstep-3]-=ptr[0];
                                        labelLab[valuePri*labstep-2]-=ptr[1];
                                        labelLab[valuePri*labstep-1]-=ptr[2];
                                        i_ori-=1;
                                        mofs=mofs-mstep;
                                    }
                                    else
                                        break;
                                }
                                else if (mask[mofs+mstep]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-1]!=valuePri&&mask[mofs-mstep]!=valuePri)
                                {
                                    if (ptrGradient[mofs]<gg||(mask[mofs+1+mstep]!=valuePri&&mask[mofs-1+mstep]!=valuePri&&(abs(ptrGrey[mofs+mstep]-ptrGrey[mofs+1+mstep])<gg||abs(ptrGrey[mofs+mstep]-ptrGrey[mofs-1+mstep])<gg)))
                                    {
                                        int del=10000;
                                        int labT;
                                        int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                        if (mask[mofs-1]>0&&del0<del)
                                        {
                                            del=del0;
                                            labT=mask[mofs-1];
                                        }
                                        del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                        if (del0<del&&mask[mofs-mstep]>0)
                                        {
                                            del=del0;
                                            labT=mask[mofs-mstep];
                                        }
                                        del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                        if (del0<del&&mask[mofs+1]>0)
                                        {
                                            //del=del0;
                                            labT=mask[mofs+1];
                                        }
                                        mask[mofs]=labT;
                                        
                                        ptrGreyMean0[labT-1]+=ptrGrey[mofs];
                                        ptrGreyMean0[valuePri-1]-=ptrGrey[mofs];
                                        ptrGreyNum0[valuePri-1]--;
                                        ptrGreyNum0[labT-1]++;
                                        ptrPosition[2*valuePri-2]=ptrPosition[2*valuePri-2]-i_ori;
                                        ptrPosition[2*valuePri-1]=ptrPosition[2*valuePri-1]-j_ori;
                                        ptrPosition[2*labT-2]=ptrPosition[2*labT-2]+i_ori;
                                        ptrPosition[2*labT-1]=ptrPosition[2*labT-1]+j_ori;
                                        uchar* ptr;
                                        ptr= img + 3*i_ori*imgcol+3*j_ori;
                                        labelLab[labT*labstep-3]+=ptr[0];
                                        labelLab[labT*labstep-2]+=ptr[1];
                                        labelLab[labT*labstep-1]+=ptr[2];
                                        labelLab[valuePri*labstep-3]-=ptr[0];
                                        labelLab[valuePri*labstep-2]-=ptr[1];
                                        labelLab[valuePri*labstep-1]-=ptr[2];
                                        i_ori+=1;
                                        mofs=mofs+mstep;
                                        //valuePri=labT;
                                    }
                                    else
                                        break;
                                }
                                else
                                    break;
                            }
                        }
                    }
                }
                
                
                
            }
            
            // mexPrintf("itr=%d\n",itr);
            
          
            
            
        
            {
                for(i=1;i<imgrow-1;i++)
                {
                    for(j=1;j < imgcol-1; j++)
                    {
                        int mofs=i*imgcol+j;
                        int valuePri=mask[mofs];
                        for(;;)
                        {
                            if (mask[mofs+1]==valuePri&&mask[mofs-1]!=valuePri&&mask[mofs-mstep]!=valuePri&&mask[mofs+mstep]!=valuePri)
                            {
                                
                                
                                if (ptrGradient[mofs]<gg||(mask[mofs+1+mstep]!=valuePri&&mask[mofs+1-mstep]!=valuePri&&(abs(ptrGrey[mofs+1]-ptrGrey[mofs+1+mstep])<lambda||abs(ptrGrey[mofs+1]-ptrGrey[mofs+1-mstep])<lambda)))
                                {
                                    int del=10000;
                                    int labT;
                                    int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                    if (mask[mofs-1]>0&&del0<del)
                                    {
                                        del=del0;
                                        labT=mask[mofs-1];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                    if (del0<del&&mask[mofs-mstep]>0)
                                    {
                                        del=del0;
                                        labT=mask[mofs-mstep];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                    if (del0<del&&mask[mofs+mstep]>0)
                                    {
                                        //del=del0;
                                        labT=mask[mofs+mstep];
                                    }
                                    mask[mofs]=labT;
                                    
                                    ptrGreyMean0[labT-1]+=ptrGrey[mofs];
                                    ptrGreyMean0[valuePri-1]-=ptrGrey[mofs];
                                    ptrGreyNum0[valuePri-1]--;
                                    ptrGreyNum0[labT-1]++;
                                    mofs=mofs+1;
                                }
                                else
                                    break;
                                
                                
                            }
                            else if (mask[mofs-1]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-mstep]!=valuePri&&mask[mofs+mstep]!=valuePri)
                            {
                                if (ptrGradient[mofs]<gg||(mask[mofs-1+mstep]!=valuePri&&mask[mofs-1-mstep]!=valuePri&&(abs(ptrGrey[mofs-1]-ptrGrey[mofs-1+mstep])<lambda||abs(ptrGrey[mofs-1]-ptrGrey[mofs-1-mstep])<lambda)))
                                {
                                    int del=10000;
                                    int labT;
                                    int del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                    if (mask[mofs+1]>0&&del0<del)
                                    {
                                        del=del0;
                                        labT=mask[mofs+1];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                    if (del0<del&&mask[mofs-mstep]>0)
                                    {
                                        del=del0;
                                        labT=mask[mofs-mstep];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                    if (del0<del&&mask[mofs+mstep]>0)
                                    {
                                        //del=del0;
                                        labT=mask[mofs+mstep];
                                    }
                                    mask[mofs]=labT;
                                    
                                    ptrGreyMean0[labT-1]+=ptrGrey[mofs];
                                    ptrGreyMean0[valuePri-1]-=ptrGrey[mofs];
                                    ptrGreyNum0[valuePri-1]--;
                                    ptrGreyNum0[labT-1]++;
                                    mofs=mofs-1;
                                }
                                else
                                    break;
                            }
                            else if (mask[mofs-mstep]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-1]!=valuePri&&mask[mofs+mstep]!=valuePri)
                            {
                                if (ptrGradient[mofs]<gg||(mask[mofs+1-mstep]!=valuePri&&mask[mofs-1-mstep]!=valuePri&&(abs(ptrGrey[mofs-mstep]-ptrGrey[mofs+1-mstep])<lambda||abs(ptrGrey[mofs-mstep]-ptrGrey[mofs-1-mstep])<lambda)))
                                {
                                    int del=10000;
                                    int labT;
                                    int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                    if (mask[mofs-1]>0&&del0<del)
                                    {
                                        del=del0;
                                        labT=mask[mofs-1];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                    if (del0<del&&mask[mofs+1]>0)
                                    {
                                        del=del0;
                                        labT=mask[mofs+1];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                    if (del0<del&&mask[mofs+mstep]>0)
                                    {
                                        //del=del0;
                                        labT=mask[mofs+mstep];
                                    }
                                    mask[mofs]=labT;
                                    
                                    ptrGreyMean0[labT-1]+=ptrGrey[mofs];
                                    ptrGreyMean0[valuePri-1]-=ptrGrey[mofs];
                                    ptrGreyNum0[valuePri-1]--;
                                    ptrGreyNum0[labT-1]++;
                                    mofs=mofs-mstep;
                                }
                                else
                                    break;
                            }
                            else if (mask[mofs+mstep]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-1]!=valuePri&&mask[mofs-mstep]!=valuePri)
                            {
                                if (ptrGradient[mofs]<gg||(mask[mofs+1+mstep]!=valuePri&&mask[mofs-1+mstep]!=valuePri&&(abs(ptrGrey[mofs+mstep]-ptrGrey[mofs+1+mstep])<lambda||abs(ptrGrey[mofs+mstep]-ptrGrey[mofs-1+mstep])<lambda)))
                                {
                                    int del=10000;
                                    int labT;
                                    int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                    if (mask[mofs-1]>0&&del0<del)
                                    {
                                        del=del0;
                                        labT=mask[mofs-1];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                    if (del0<del&&mask[mofs-mstep]>0)
                                    {
                                        del=del0;
                                        labT=mask[mofs-mstep];
                                    }
                                    del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                    if (del0<del&&mask[mofs+1]>0)
                                    {
                                        //del=del0;
                                        labT=mask[mofs+1];
                                    }
                                    mask[mofs]=labT;
                                    
                                    ptrGreyMean0[labT-1]+=ptrGrey[mofs];
                                    ptrGreyMean0[valuePri-1]-=ptrGrey[mofs];
                                    ptrGreyNum0[valuePri-1]--;
                                    ptrGreyNum0[labT-1]++;
                                    mofs=mofs+mstep;
                                    //valuePri=labT;
                                }
                                else
                                    break;
                            }
                            else
                                break;
                        }
                    }
                }
            }
            cvReleaseMat(&cvPosition);
            
            
            
            
           
            cvReleaseMat(&cvGreyMean0);
            cvReleaseMat(&cvGreyNum0);
           
            if (SM==1)
                
            {
                //local marching
                int itrr=2;
                for(int rr=1;rr<itrr+1;rr++)
                {
                    cv::Mat PositionMat0=Mat::zeros(labelId, 2, CV_32SC1);
                    CvMat *cvPosition0 = cvCreateMat(labelId,2,CV_32SC1);
                    int *ptrPosition0;
                    CvMat Position0=PositionMat0;
                    cvCopy(& Position0, cvPosition0);
                    ptrPosition0=cvPosition0->data.i;
                    cv::Mat NumberMat0=Mat::zeros(labelId, 1, CV_32SC1);
                    CvMat *cvNumber0 = cvCreateMat(labelId,1,CV_32SC1);
                    int *ptrNumber0;
                    CvMat Number0=NumberMat0;
                    cvCopy(& Number0, cvNumber0);
                    ptrNumber0=cvNumber0->data.i;
                    
                    
                    cv::Mat SNeighborMat0=Mat::zeros(labelId, labelId, CV_32SC1);
                    CvMat *cvSNeighbor0 = cvCreateMat(labelId,labelId,CV_32SC1);
                    int *ptrSNeighbor0;
                    CvMat SNeighbor0=SNeighborMat0;
                    cvCopy(& SNeighbor0, cvSNeighbor0);
                    ptrSNeighbor0=cvSNeighbor0->data.i;
                    
                    cv::Mat StdMat0=Mat::zeros(labelId, 1, CV_32SC1);
                    CvMat *cvStd0 = cvCreateMat(labelId,1,CV_32SC1);
                    int *ptrStd0;
                    CvMat Std0=StdMat0;
                    cvCopy(& Std0, cvStd0);
                    ptrStd0=cvStd0->data.i;
                    
                    cv::Mat SNumberMat0=Mat::zeros(labelId, labelId, CV_32SC1);
                    CvMat *cvSNumber0 = cvCreateMat(labelId,labelId,CV_32SC1);
                    int *ptrSNumber0;
                    CvMat SNumber0=SNumberMat0;
                    cvCopy(& SNumber0, cvSNumber0);
                    ptrSNumber0=cvSNumber0->data.i;
                    
                    cv::Mat SMeanColorMat0=Mat::zeros(labelId, labelId, CV_32SC3);
                    CvMat *cvSMeanColor0 = cvCreateMat(labelId,labelId,CV_32SC3);
                    int *ptrSMeanColor0;
                    CvMat SMeanColor0=SMeanColorMat0;
                    cvCopy(& SMeanColor0, cvSMeanColor0);
                    ptrSMeanColor0=cvSMeanColor0->data.i;
                    
                    cv::Mat TextureMat0=Mat::zeros(labelId, 1, CV_32SC1);
                    CvMat *cvTexture0 = cvCreateMat(labelId,1,CV_32SC1);
                    int *ptrTexture0;
                    CvMat Texture0=TextureMat0;
                    cvCopy(& Texture0, cvTexture0);
                    ptrTexture0=cvTexture0->data.i;
                    
                    
                    for (i=0;i<labelId;i++)
                    {
                        labelLab[3*i]=0;
                        labelLab[3*i+1]=0;
                        labelLab[3*i+2]=0;
                    }
                    
                    //累计各SP的中心位置等
                    for(i=1;i<imgrow-1;i++)
                    {
                        for(j=1;j < imgcol-1; j++)
                        {
                            int mm=i*imgcol+j;
                            int labelpri=mask[mm];
                            ptrNumber0[labelpri-1]++;
                            ptrPosition0[2*labelpri-2]+=i;
                            ptrPosition0[2*labelpri-1]+=j;
                            ptrTexture0[labelpri-1]+=ptrimgGradient[mm];
                            uchar* ptr=img+3*mm;
                            labelLab[labelpri*labstep-3]+=ptr[0];
                            labelLab[labelpri*labstep-2]+=ptr[1];
                            labelLab[labelpri*labstep-1]+=ptr[2];
                        }
                    }
                    
                    
                    for(i=0;i<imgrow;i++)
                    {
                        for(j=0;j < imgcol ; j++)
                        {
                            iflabeled[i*imgcol+j]=0;
                        }
                    }
                    for(i=0;i<imgrow;i++)
                    {
                        iflabeled[i*imgcol]=-200;
                        iflabeled[i*imgcol+imgcol-1]=-200;
                    }
                    for(j=0;j < imgcol; j++)
                    {
                        iflabeled[j]=-200;
                        iflabeled[imgrow*imgcol-imgcol+j]=-200;
                    }
                    
                    
                    int gth=5;
                    double cth=3;
                    double ccth=cth;
                    double tth=20;
                    double sth=2;
                    gg=30;
                    
                    int SQ=ceil(disAver/2);
                    
                    
                    
                    for(i=1;i<imgrow-1;i++)
                    {
                        for(j=1;j < imgcol-1; j++)
                        {
                            int* m;
                            //m=mask+i*imgcol+j;
                            int mofs=i*imgcol+j;
                            m=mask+mofs;
                            int labelpri=m[0];
                            uchar* ptr=img+3*mofs;
                            if (m[1]!=labelpri&&m[1]>0)
                            {
                                int labelnew=m[1];
                                ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]+=ptrGradient[i*imgcol+j];
                                ptrSNumber0[(labelpri-1)*labelId+labelnew-1]+=1;
                                ptrSMeanColor0[3*(labelpri-1)*labelId+3*labelnew-3]+=ptr[0];
                                ptrSMeanColor0[3*(labelpri-1)*labelId+3*labelnew-2]+=ptr[1];
                                ptrSMeanColor0[3*(labelpri-1)*labelId+3*labelnew-1]+=ptr[2];
                            }
                            else if (m[-1]!=labelpri&&m[-1]>0)
                            {
                                int labelnew=m[-1];
                                ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]+=ptrGradient[i*imgcol+j];
                                ptrSNumber0[(labelpri-1)*labelId+labelnew-1]+=1;
                                ptrSMeanColor0[3*(labelpri-1)*labelId+3*labelnew-3]+=ptr[0];
                                ptrSMeanColor0[3*(labelpri-1)*labelId+3*labelnew-2]+=ptr[1];
                                ptrSMeanColor0[3*(labelpri-1)*labelId+3*labelnew-1]+=ptr[2];
                            }
                            else if (m[mstep]!=labelpri&&m[mstep]>0)
                            {
                                int labelnew=m[mstep];
                                ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]+=ptrGradient[i*imgcol+j];
                                ptrSNumber0[(labelpri-1)*labelId+labelnew-1]+=1;
                                ptrSMeanColor0[3*(labelpri-1)*labelId+3*labelnew-3]+=ptr[0];
                                ptrSMeanColor0[3*(labelpri-1)*labelId+3*labelnew-2]+=ptr[1];
                                ptrSMeanColor0[3*(labelpri-1)*labelId+3*labelnew-1]+=ptr[2];
                            }
                            else if (m[-mstep]!=labelpri&&m[-mstep]>0)
                            {
                                int labelnew=m[-mstep];
                                ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]+=ptrGradient[i*imgcol+j];
                                ptrSNumber0[(labelpri-1)*labelId+labelnew-1]+=1;
                                ptrSMeanColor0[3*(labelpri-1)*labelId+3*labelnew-3]+=ptr[0];
                                ptrSMeanColor0[3*(labelpri-1)*labelId+3*labelnew-2]+=ptr[1];
                                ptrSMeanColor0[3*(labelpri-1)*labelId+3*labelnew-1]+=ptr[2];
                            }
                            
                            
                            
                        }
                    }
                    //计算texture的方差
                    for(i=1;i<imgrow-1;i++)
                    {
                        for(j=1;j < imgcol-1; j++)
                        {
                            
                            int mofs=i*imgcol+j;
                            
                            int labelpri=mask[mofs];
                            int meanValue=ptrTexture0[labelpri-1]/ptrNumber0[labelpri-1];
                            int temp=abs(ptrimgGradient[mofs]-meanValue);
                            ptrStd0[labelpri-1]+=temp*temp;
                        }
                    }
                    for(i=0;i<labelId;i++)
                    {
                        if (ptrNumber0[i]!=0)
                            ptrStd0[i]=ptrStd0[i]/ptrNumber0[i];
                    }
                    

                    //筛选可以local marching的边界，可以为1，不可以为0
                    int numCount=0;
                    double minC=500;
                    double maxC=0;
                    for(i=0;i<labelId;i++)
                    {
                        for(j=0;j < labelId; j++)
                        {
                            if (ptrSNumber0[i*labelId+j]>0)
                            {
                                int mofs=i*labelId+j;
                                int tempvalue=0;
                                
                                ptrSNeighbor0[mofs]=ptrSNeighbor0[mofs]/ptrSNumber0[mofs];
                                
                                double lColor1=double(labelLab[i*labstep])/double(ptrNumber0[i]);
                                double aColor1=double(labelLab[i*labstep+1])/double(ptrNumber0[i]);
                                double  bColor1=double(labelLab[i*labstep+2])/double(ptrNumber0[i]);
                                double   lColor2=double(labelLab[j*labstep])/double(ptrNumber0[j]);
                                double   aColor2=double(labelLab[j*labstep+1])/double(ptrNumber0[j]);
                                double    bColor2=double(labelLab[j*labstep+2])/double(ptrNumber0[j]);
                                double    diffC= 0.1*(lColor1-lColor2)*(lColor1-lColor2)+1.45*(aColor1-aColor2)*(aColor1-aColor2)+1.45*(bColor1-bColor2)*(bColor1-bColor2);
                                if (ptrSNeighbor0[mofs]>gth)
                                {
                                    
                                    lColor1=double(ptrSMeanColor0[3*mofs])/double(ptrSNumber0[mofs]);
                                    aColor1=double(ptrSMeanColor0[3*mofs+1])/double(ptrSNumber0[mofs]);
                                    bColor1=double(ptrSMeanColor0[3*mofs+2])/double(ptrSNumber0[mofs]);
                                    lColor2=double(labelLab[j*labstep])/double(ptrNumber0[j]);
                                    aColor2=double(labelLab[j*labstep+1])/double(ptrNumber0[j]);
                                    bColor2=double(labelLab[j*labstep+2])/double(ptrNumber0[j]);
                                    
                                    double diffCE;
                                    diffCE= 0.1*(lColor1-lColor2)*(lColor1-lColor2)+1.45*(aColor1-aColor2)*(aColor1-aColor2)+1.45*(bColor1-bColor2)*(bColor1-bColor2);
                                    
                                    if (sqrt(diffCE)<=cth&&ptrSNeighbor0[mofs]<=3*gth)
                                    {
                                        tempvalue=1;
                                    }
                                    
                                    
                                    if (ptrSNeighbor0[mofs]<=4*gth&&tempvalue==0)
                                    {
                                                
                                        double texture1= double(ptrTexture0[i])/double(ptrNumber0[i]);
                                        double texture2= double(ptrTexture0[j])/double(ptrNumber0[j]);
                                        
                                        if (sqrt(diffC)<ccth&&abs(texture1-texture2)<=tth&&sqrt(ptrStd0[i])>=sth&&sqrt(ptrStd0[j])>=sth)
                                            
                                        {
                                            tempvalue=1;
                                        }
                                        
                                    }
                                }
                                else
                                {
                                    tempvalue=1;
                                }
                                ptrSNeighbor0[mofs]=tempvalue;
                            }
                        }
                    }
                    
                    
                    int valuePri;
                    int labelpri;
                    int tagL=0-rr;
                    active_queue=0;
                    
                    for(i=1;i<imgrow-1;i++)
                    {
                        for(j=1;j < imgcol-1; j++)
                        {
                            int mofs;
                            int* m;
                            m=mask+i*imgcol+j;
                            labelpri=m[0];
                            int mp=i*imgcol+j;
                            mofs=mp;
                            valuePri=labelpri;
                            
                            
                            if (m[1]>labelpri&&m[1]>0&&iflabeled[mp]!=tagL)
                            {
                                ws_push(0,mp, i, j);
                                iflabeled[mp]=tagL;
                            }
                            else if (m[-1]>labelpri&&m[-1]>0&&iflabeled[mp]!=tagL)
                            {
                                ws_push(0,mp, i, j);
                                iflabeled[mp]=tagL;
                            }
                            else if (m[mstep]>labelpri&&m[mstep]>0&&iflabeled[mp]!=tagL)
                            {
                                ws_push(0,mp, i, j);
                                iflabeled[mp]=tagL;
                            }
                            else if (m[-mstep]>labelpri&&m[-mstep]>0&&iflabeled[mp]!=tagL)
                            {
                                ws_push(0,mp, i, j);
                                iflabeled[mp]=tagL;
                            }
                        }
                    }
                    
                    
                    for (;;)
                    {
                        if (q[active_queue].first == 0)
                        {
                            int qqq;
                            for (qqq = active_queue+1; qqq < SQ+1; qqq++)
                                if (q[qqq].first)
                                    break;
                            if (qqq > SQ)
                                break;
                            active_queue = qqq;
                        }
                        int i_ori,j_ori;
                        int mofs;
                        ws_pop(active_queue, mofs, i_ori, j_ori);
                        int valuePri=mask[mofs];
                        int labelpri=valuePri;
                        

                        iflabeled[mofs]=0;
                        
                        
                        if (mask[mofs+1]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs+mstep]!=valuePri&&(mask[mofs-mstep]!=valuePri||mask[mofs-mstep+1]!=valuePri||mask[mofs-mstep-1]!=valuePri))
                        {
                            // iflabeled[mofs]=0;
                        }
                        else if (mask[mofs+1]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs-mstep]!=valuePri&&(mask[mofs+mstep]!=valuePri||mask[mofs+mstep+1]!=valuePri||mask[mofs+mstep-1]!=valuePri))
                        {
                            // iflabeled[mofs]=0;
                        }
                        else if (mask[mofs+mstep]==valuePri&&mask[mofs-mstep]==valuePri&&mask[mofs-1]!=valuePri&&(mask[mofs+mstep+1]!=valuePri||mask[mofs-mstep+1]!=valuePri||mask[mofs+1]!=valuePri))
                        {
                            // iflabeled[mofs]=0;
                        }
                        else if (mask[mofs+mstep]==valuePri&&mask[mofs-mstep]==valuePri&&mask[mofs+1]!=valuePri&&(mask[mofs+mstep-1]!=valuePri||mask[mofs-mstep-1]!=valuePri||mask[mofs-1]!=valuePri))
                        {
                            //  iflabeled[mofs]=0;
                        }
                        else if (mask[mofs+mstep]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs+mstep-1]!=valuePri)
                        {
                            // iflabeled[mofs]=0;
                        }
                        else if (mask[mofs+mstep]==valuePri&&mask[mofs+1]==valuePri&&mask[mofs+mstep+1]!=valuePri)
                        {
                            //  iflabeled[mofs]=0;
                        }
                        else if (mask[mofs-mstep]==valuePri&&mask[mofs+1]==valuePri&&mask[mofs-mstep+1]!=valuePri)
                        {
                            //  iflabeled[mofs]=0;
                        }
                        else if (mask[mofs-mstep]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs-mstep-1]!=valuePri)
                        {
                            //iflabeled[mofs]=0;
                        }
                        //if(0<1)
                        else
                        {
                            double minT=100000000000000000;
                            int labT=labelpri;
                            int flag=0;
                            //可以移动的边界开始local marching
                            if (mask[mofs+mstep]!=labelpri&&mask[mofs+mstep]>0)
                            {
                                int labelnew=mask[mofs+mstep];
                                if (ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]==1&&ptrSNeighbor0[(labelnew-1)*labelId+labelpri-1]==1)
                                {
                                    flag=1;
                                    double i_pri=double(ptrPosition0[2*labelpri-2])/double(ptrNumber0[labelpri-1]);
                                    double j_pri=double(ptrPosition0[2*labelpri-1])/double(ptrNumber0[labelpri-1]);
                                    double i_new=double(ptrPosition0[2*labelnew-2])/double(ptrNumber0[labelnew-1]);
                                    double j_new=double(ptrPosition0[2*labelnew-1])/double(ptrNumber0[labelnew-1]);
                                    double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                    if (disT<(i_ori-i_pri)*(i_ori-i_pri)+(j_ori-j_pri)*(j_ori-j_pri)&&disT<minT)
                                    {
                                        minT=disT;
                                        labT=labelnew;
                                    }
                                }
                                
                            }
                            if (mask[mofs-mstep]!=labelpri&&mask[mofs-mstep]>0)
                            {
                                int labelnew=mask[mofs-mstep];
                                if (ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]==1&&ptrSNeighbor0[(labelnew-1)*labelId+labelpri-1]==1)
                                {
                                    flag=1;
                                    double i_pri=double(ptrPosition0[2*labelpri-2])/double(ptrNumber0[labelpri-1]);
                                    double j_pri=double(ptrPosition0[2*labelpri-1])/double(ptrNumber0[labelpri-1]);
                                    double i_new=double(ptrPosition0[2*labelnew-2])/double(ptrNumber0[labelnew-1]);
                                    double j_new=double(ptrPosition0[2*labelnew-1])/double(ptrNumber0[labelnew-1]);
                                    double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                    if (disT<(i_ori-i_pri)*(i_ori-i_pri)+(j_ori-j_pri)*(j_ori-j_pri)&&disT<minT)
                                    {
                                        minT=disT;
                                        labT=labelnew;
                                    }
                                }
                            }
                            if (mask[mofs+1]!=labelpri&&mask[mofs+1]>0)
                            {
                                int labelnew=mask[mofs+1];
                                
                                if (ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]==1&&ptrSNeighbor0[(labelnew-1)*labelId+labelpri-1]==1)
                                {
                                    flag=1;
                                    double i_pri=double(ptrPosition0[2*labelpri-2])/double(ptrNumber0[labelpri-1]);
                                    double j_pri=double(ptrPosition0[2*labelpri-1])/double(ptrNumber0[labelpri-1]);
                                    double i_new=double(ptrPosition0[2*labelnew-2])/double(ptrNumber0[labelnew-1]);
                                    double j_new=double(ptrPosition0[2*labelnew-1])/double(ptrNumber0[labelnew-1]);
                                    double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                    if (disT<(i_ori-i_pri)*(i_ori-i_pri)+(j_ori-j_pri)*(j_ori-j_pri)&&disT<minT)
                                    {
                                        minT=disT;
                                        labT=labelnew;
                                    }
                                }
                            }
                            if (mask[mofs-1]!=labelpri&&mask[mofs-1]>0)
                            {
                                int labelnew=mask[mofs-1];
                                if (ptrSNeighbor0[(labelpri-1)*labelId+labelnew-1]==1&&ptrSNeighbor0[(labelnew-1)*labelId+labelpri-1]==1)
                                {
                                    flag=1;
                                    double i_pri=double(ptrPosition0[2*labelpri-2])/double(ptrNumber0[labelpri-1]);
                                    double j_pri=double(ptrPosition0[2*labelpri-1])/double(ptrNumber0[labelpri-1]);
                                    double i_new=double(ptrPosition0[2*labelnew-2])/double(ptrNumber0[labelnew-1]);
                                    double j_new=double(ptrPosition0[2*labelnew-1])/double(ptrNumber0[labelnew-1]);
                                    double disT=(i_ori-i_new)*(i_ori-i_new)+(j_ori-j_new)*(j_ori-j_new);
                                    if (disT<(i_ori-i_pri)*(i_ori-i_pri)+(j_ori-j_pri)*(j_ori-j_pri)&&disT<minT)
                                    {
                                        minT=disT;
                                        labT=labelnew;
                                    }
                                }
                            }
                            
                            if (labT!=labelpri)
                            {
                                mask[mofs]=labT;
                                ptrNumber0[labelpri-1]--;
                                ptrNumber0[labT-1]++;
                                ptrPosition0[2*labelpri-2]-=i_ori;
                                ptrPosition0[2*labelpri-1]-=j_ori;
                                ptrPosition0[2*labT-2]+=i_ori;
                                ptrPosition0[2*labT-1]+=j_ori;
                                uchar* ptr;
                                ptr= img + 3*i_ori*imgcol+3*j_ori;
                                labelLab[labT*labstep-3]+=ptr[0];
                                labelLab[labT*labstep-2]+=ptr[1];
                                labelLab[labT*labstep-1]+=ptr[2];
                                labelLab[labelpri*labstep-3]-=ptr[0];
                                labelLab[labelpri*labstep-2]-=ptr[1];
                                labelLab[labelpri*labstep-1]-=ptr[2];
                                ptrTexture0[labelpri-1]-=ptrimgGradient[mofs];
                                ptrTexture0[labT-1]+=ptrimgGradient[mofs];
                                
                                labelpri=labT;
                            }
                        }
                        if(active_queue<SQ-1)
                        {
                            if (mask[mofs+1]!=labelpri&&mask[mofs+1]>0&&iflabeled[mofs+1]==0)
                            {
                                ws_push(active_queue+1,mofs+1, i_ori, j_ori+1);
                                
                                iflabeled[mofs+1]=tagL;
                            }
                            if (mask[mofs-1]!=labelpri&&mask[mofs-1]>0&&iflabeled[mofs-1]==0)
                            {
                                ws_push(active_queue+1,mofs-1, i_ori, j_ori-1);
                                iflabeled[mofs-1]=tagL;
                            }
                            if (mask[mofs+mstep]!=labelpri&&mask[mofs+mstep]>0&&iflabeled[mofs+mstep]==0)
                            {
                                ws_push(active_queue+1,mofs+mstep, i_ori+1, j_ori);
                                iflabeled[mofs+mstep]=tagL;
                            }
                            if (mask[mofs-mstep]!=labelpri&&mask[mofs-mstep]>0&&iflabeled[mofs-mstep]==0)
                            {
                                ws_push(active_queue+1,mofs-mstep, i_ori-1, j_ori);
                                iflabeled[mofs-mstep]=tagL;
                            }
                        }
                        
                        
            
                        
                    }
                    
                   
                    {
                        for(i=1;i<imgrow-1;i++)
                        {
                            for(j=1;j < imgcol-1; j++)
                            {
                                int mofs=i*imgcol+j;
                                int valuePri=mask[mofs];
                                if (mask[mofs+1]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs+mstep]!=valuePri&&(mask[mofs-mstep]!=valuePri||mask[mofs-mstep+1]!=valuePri||mask[mofs-mstep-1]!=valuePri))
                                {
                                    //iflabeled[mofs]=0;
                                }
                                else if (mask[mofs+1]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs-mstep]!=valuePri&&(mask[mofs+mstep]!=valuePri||mask[mofs+mstep+1]!=valuePri||mask[mofs+mstep-1]!=valuePri))
                                {
                                    // iflabeled[mofs]=0;
                                }
                                else if (mask[mofs+mstep]==valuePri&&mask[mofs-mstep]==valuePri&&mask[mofs-1]!=valuePri&&(mask[mofs+mstep+1]!=valuePri||mask[mofs-mstep+1]!=valuePri||mask[mofs+1]!=valuePri))
                                {
                                    //  iflabeled[mofs]=0;
                                }
                                else if (mask[mofs+mstep]==valuePri&&mask[mofs-mstep]==valuePri&&mask[mofs+1]!=valuePri&&(mask[mofs+mstep-1]!=valuePri||mask[mofs-mstep-1]!=valuePri||mask[mofs-1]!=valuePri))
                                {
                                    // iflabeled[mofs]=0;
                                }
                                else if (mask[mofs+mstep]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs+mstep-1]!=valuePri)
                                {
                                    //  iflabeled[mofs]=0;
                                }
                                else if (mask[mofs+mstep]==valuePri&&mask[mofs+1]==valuePri&&mask[mofs+mstep+1]!=valuePri)
                                {
                                    // iflabeled[mofs]=0;
                                }
                                else if (mask[mofs-mstep]==valuePri&&mask[mofs+1]==valuePri&&mask[mofs-mstep+1]!=valuePri)
                                {
                                    // iflabeled[mofs]=0;
                                }
                                else if (mask[mofs-mstep]==valuePri&&mask[mofs-1]==valuePri&&mask[mofs-mstep-1]!=valuePri)
                                {
                                    // iflabeled[mofs]=0;
                                }
                                else
                                    for(;;)
                                    {
                                        if (mask[mofs+1]==valuePri&&mask[mofs-1]!=valuePri&&mask[mofs-mstep]!=valuePri&&mask[mofs+mstep]!=valuePri)
                                        {
                                            
                                            
                                            if (ptrGradient[mofs]<gg||(mask[mofs+1+mstep]!=valuePri&&mask[mofs+1-mstep]!=valuePri&&(abs(ptrGrey[mofs+1]-ptrGrey[mofs+1+mstep])<lambda||abs(ptrGrey[mofs+1]-ptrGrey[mofs+1-mstep])<lambda)))
                                            {
                                                int del=10000;
                                                int labT;
                                                int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                                if (mask[mofs-1]>0&&del0<del)
                                                {
                                                    del=del0;
                                                    labT=mask[mofs-1];
                                                }
                                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                                if (del0<del&&mask[mofs-mstep]>0)
                                                {
                                                    del=del0;
                                                    labT=mask[mofs-mstep];
                                                }
                                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                                if (del0<del&&mask[mofs+mstep]>0)
                                                {
                                                    //del=del0;
                                                    labT=mask[mofs+mstep];
                                                }
                                                mask[mofs]=labT;
                                                
                                                mofs=mofs+1;
                                            }
                                            else
                                                break;
                                            
                                            
                                        }
                                        else if (mask[mofs-1]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-mstep]!=valuePri&&mask[mofs+mstep]!=valuePri)
                                        {
                                            if (ptrGradient[mofs]<gg||(mask[mofs-1+mstep]!=valuePri&&mask[mofs-1-mstep]!=valuePri&&(abs(ptrGrey[mofs-1]-ptrGrey[mofs-1+mstep])<lambda||abs(ptrGrey[mofs-1]-ptrGrey[mofs-1-mstep])<lambda)))
                                            {
                                                int del=10000;
                                                int labT;
                                                int del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                                if (mask[mofs+1]>0&&del0<del)
                                                {
                                                    del=del0;
                                                    labT=mask[mofs+1];
                                                }
                                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                                if (del0<del&&mask[mofs-mstep]>0)
                                                {
                                                    del=del0;
                                                    labT=mask[mofs-mstep];
                                                }
                                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                                if (del0<del&&mask[mofs+mstep]>0)
                                                {
                                                    //del=del0;
                                                    labT=mask[mofs+mstep];
                                                }
                                                mask[mofs]=labT;
                                                
                                                mofs=mofs-1;
                                            }
                                            else
                                                break;
                                        }
                                        else if (mask[mofs-mstep]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-1]!=valuePri&&mask[mofs+mstep]!=valuePri)
                                        {
                                            if (ptrGradient[mofs]<gg||(mask[mofs+1-mstep]!=valuePri&&mask[mofs-1-mstep]!=valuePri&&(abs(ptrGrey[mofs-mstep]-ptrGrey[mofs+1-mstep])<lambda||abs(ptrGrey[mofs-mstep]-ptrGrey[mofs-1-mstep])<lambda)))
                                            {
                                                int del=10000;
                                                int labT;
                                                int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                                if (mask[mofs-1]>0&&del0<del)
                                                {
                                                    del=del0;
                                                    labT=mask[mofs-1];
                                                }
                                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                                if (del0<del&&mask[mofs+1]>0)
                                                {
                                                    del=del0;
                                                    labT=mask[mofs+1];
                                                }
                                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs+mstep]);
                                                if (del0<del&&mask[mofs+mstep]>0)
                                                {
                                                    //del=del0;
                                                    labT=mask[mofs+mstep];
                                                }
                                                mask[mofs]=labT;
                                                
                                                mofs=mofs-mstep;
                                            }
                                            else
                                                break;
                                        }
                                        else if (mask[mofs+mstep]==valuePri&&mask[mofs+1]!=valuePri&&mask[mofs-1]!=valuePri&&mask[mofs-mstep]!=valuePri)
                                        {
                                            if (ptrGradient[mofs]<gg||(mask[mofs+1+mstep]!=valuePri&&mask[mofs-1+mstep]!=valuePri&&(abs(ptrGrey[mofs+mstep]-ptrGrey[mofs+1+mstep])<lambda||abs(ptrGrey[mofs+mstep]-ptrGrey[mofs-1+mstep])<lambda)))
                                            {
                                                int del=10000;
                                                int labT;
                                                int del0=abs(ptrGrey[mofs]-ptrGrey[mofs-1]);
                                                if (mask[mofs-1]>0&&del0<del)
                                                {
                                                    del=del0;
                                                    labT=mask[mofs-1];
                                                }
                                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs-mstep]);
                                                if (del0<del&&mask[mofs-mstep]>0)
                                                {
                                                    del=del0;
                                                    labT=mask[mofs-mstep];
                                                }
                                                del0=abs(ptrGrey[mofs]-ptrGrey[mofs+1]);
                                                if (del0<del&&mask[mofs+1]>0)
                                                {
                                                    //del=del0;
                                                    labT=mask[mofs+1];
                                                }
                                                mask[mofs]=labT;
                                                
                                                mofs=mofs+mstep;
                                                //valuePri=labT;
                                            }
                                            else
                                                break;
                                        }
                                        else
                                            break;
                                    }
                                
                            }
                        }
                    }
                    
                    cvReleaseMat(&cvSNeighbor0);
                    cvReleaseMat(&cvSNumber0);
                    cvReleaseMat(&cvPosition0);
                    cvReleaseMat(&cvNumber0);
                    cvReleaseMat(&cvStd0);
                    cvReleaseMat(&cvSMeanColor0);
                    cvReleaseMat(&cvTexture0);
                }
                
                
                
            }
            
            
            
            
            
            
            
            
        }
        
    
        
        cvReleaseMat(&cvGrey);
        
    }
   
    labelnumber[0]=labelId;

    for(i=0;i<imgrow;i++)
    {
        mask[i*imgcol]=mask[i*imgcol+1];
        mask[i*imgcol+imgcol-1]=mask[i*imgcol+imgcol-2];
    }
    for(j=0;j < imgcol; j++)
    {
        mask[j]=mask[j+mstep];
        mask[imgrow*imgcol-imgcol+j]=mask[imgrow*imgcol-imgcol+j-mstep];
    }
    mask[0]=mask[mstep+1];
    mask[imgrow*imgcol-1]=mask[imgrow*imgcol-2-mstep];
    mask[imgcol-1]=mask[imgcol+mstep-2];
    mask[imgrow*imgcol-imgcol]=mask[imgrow*imgcol-imgcol+1-mstep];
    BMat = Mat(dst, true);

    
    
    
    
    cvReleaseMat(&src);
    cvReleaseMat(&dst);
    cvReleaseMat(&labellabptr);
    cvReleaseMat(&numberlabptr);
    cvReleaseMat(&iflabel);
}




void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    // ============ parse input ==============
    // create opencv Mat from argument
    
    
    cv::Mat I;
    convertMx2Mat(prhs[0], I);
    int spn = (int)(mxGetScalar(prhs[1])); // number of segments
    int SM = (int)(mxGetScalar(prhs[2]));
    int ItrSet= (int)(mxGetScalar(prhs[3]));
    int lambda = (int)(mxGetScalar(prhs[4]));
    
    
    Mat seeds;
    
    
    
    
    int * labelnumber;
    labelnumber=new int[1];
    // ================== process ================
    cv::Mat B;
    compact_watershed(I, B, seeds, spn, ItrSet,SM, lambda,labelnumber);

    // ================ create output ================
    
    
    
    
    
    
    if( nlhs>0)
    {
        convertMat2Mx(B, plhs[0]);
    }
    
    int* spnumber;
    plhs[1] = mxCreateNumericMatrix(1,1,mxINT32_CLASS,mxREAL);
    
    
    
    spnumber = (int*)mxGetData(plhs[1]);//gives a void*, cast it to int*
    
    *spnumber=labelnumber[0];
    
    
    
    
}

//// mexOpenCV mex_SCAC.cpp mex_helper_SCAC.cpp