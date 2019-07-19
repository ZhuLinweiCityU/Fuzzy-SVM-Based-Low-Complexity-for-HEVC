/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2015, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     TAppEncTop.cpp
    \brief    Encoder application class
*/

#include <list>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <assert.h>
#include <iomanip>

#include "TAppEncTop.h"
#include "TLibEncoder/AnnexBwrite.h"
#include "TLibCommon/svm.h"

using namespace std;

// #include <opencv2/opencv.hpp>
// #include <opencv2/highgui//highgui.hpp>
// #include <opencv2/ml/ml.hpp>
// using namespace cv;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
//! \ingroup TAppEncoder
//! \{

ofstream f0("data0.txt"/*,ios::app*/);
ofstream f1("data1.txt"/*,ios::app*/);
ofstream f2("data2.txt"/*,ios::app*/);
//ofstream f3("data3.txt");
// ====================================================================================================================
// Constructor / destructor / initialization / destroy
// ====================================================================================================================

TAppEncTop::TAppEncTop()
{
  m_iFrameRcvd = 0;
  m_totalBytes = 0;
  m_essentialBytes = 0;
}

TAppEncTop::~TAppEncTop()
{
}

Void TAppEncTop::SVM_Train_Online(Double *Feature0, Double *Feature1, Double *Feature2,
			   Int *Truth0, Int *Truth1, Int *Truth2,
			   maxmin *M0,maxmin *M1,maxmin *M2,Int Feature_Num_Level0, Int Feature_Num_Level1, Int Feature_Num_Level2, Int trainingnum,
			   Int frameSize0, Int frameSize1, Int frameSize2,
			   Int W0,Int W1, Int W2,Int WIDTH, Int HEIGHT,
			   svm_model *&model0,svm_model*&model1,svm_model*&model2)
{
	struct svm_parameter param;		// set by parse_command_line
	struct svm_problem prob0;		// set by read_problem
	struct svm_problem prob1;		// set by read_problem
	struct svm_problem prob2;		// set by read_problem
	struct svm_node *data0,*data1,*data2;

	param.svm_type=C_SVC;
	param.C=1;
	param.degree=3;
	param.kernel_type=RBF;
	param.eps=1e-5;
	param.gamma=0.5;

	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 2;
	param.weight_label = new int[param.nr_weight];
	param.weight_label[0] = -1;
	param.weight_label[1] = +1;
	param.weight = new double[param.nr_weight];
	param.weight[0] = 1;
	param.weight[1] = 1;//zhulinwei
	//--------------------------------------------------------------------------
	for (int j=0;j<Feature_Num_Level0;j++)
	{
		M0[j].maxvalue = Feature0[j];
		M0[j].minvalue = Feature0[j];
	}
	for (int j=0;j<Feature_Num_Level1;j++)
	{
		M1[j].maxvalue = Feature1[j];
		M1[j].minvalue = Feature1[j];
	}
	for (int j=0;j<Feature_Num_Level2;j++)
	{
		M2[j].maxvalue = Feature2[j];
		M2[j].minvalue = Feature2[j];
	}


	int W00=WIDTH/64,H00=HEIGHT/64;
	int W11=WIDTH/32,H11=HEIGHT/32;
	int W22=WIDTH/16,H22=HEIGHT/16;

	for (int i=0;i<trainingnum;i++)
	{
		for (int j=0;j<frameSize0;j++)
		{
			if (0 != Truth0[i*frameSize0+j]&&(j/W0<H00&&j%W0<W00))
			{
				for (int k=0;k<Feature_Num_Level0;k++)
				{
					if (Feature0[(i*frameSize0+j)*Feature_Num_Level0+k]>=M0[k].maxvalue)
						M0[k].maxvalue = Feature0[(i*frameSize0+j)*Feature_Num_Level0+k];
					else if(Feature0[(i*frameSize0+j)*Feature_Num_Level0+k]<=M0[k].minvalue)
						M0[k].minvalue = Feature0[(i*frameSize0+j)*Feature_Num_Level0+k];
				}
			}
		}
	}

	for (int frame=0;frame<trainingnum;frame++)
	{
		for (int i=0;i<frameSize0;i++)
		{
			int tempx=2*(i/W0),tempy=2*(i%W0);
			for (int l=0;l<4;l++)
			{
				int temp1=W1*(tempx+l/2)+(tempy+l%2);
				if(0!=Truth1[4*i+l+frame*frameSize1]&&(temp1/W1<H11&&temp1%W1<W11)) 
				{
					for (int j=0;j<Feature_Num_Level1;j++)
					{
						if (Feature1[(frame*frameSize1+temp1)*Feature_Num_Level1+j]>=M1[j].maxvalue)     M1[j].maxvalue=Feature1[(frame*frameSize1+temp1)*Feature_Num_Level1+j];
						else if(Feature1[(frame*frameSize1+temp1)*Feature_Num_Level1+j]<=M1[j].minvalue) M1[j].minvalue=Feature1[(frame*frameSize1+temp1)*Feature_Num_Level1+j];
					}
				}
			}
		}
	}

	for (int frame=0;frame<trainingnum;frame++)
	{
		for (int i=0;i<frameSize0;i++)
		{
			int tempx=2*(i/W0),tempy=2*(i%W0);
			for (int l=0;l<4;l++)
			{
				int temp1=W1*(tempx+l/2)+(tempy+l%2);
				int tempxx=2*(temp1/W1), tempyy=2*(temp1%W1);
				for (int m=0;m<4;m++)
				{
					int temp2=W2*(tempxx+m/2)+(tempyy+m%2);
					if(0!=Truth2[4*(4*i+l)+m+frame*frameSize2]&&(temp2/W2<H22&&temp2%W2<W22)) 
					{
						for (int j=0;j<Feature_Num_Level2;j++)
						{
							if (Feature2[(frame*frameSize2+temp2)*Feature_Num_Level2+j]>=M2[j].maxvalue)     M2[j].maxvalue=Feature2[(frame*frameSize2+temp2)*Feature_Num_Level2+j];
							else if(Feature2[(frame*frameSize2+temp2)*Feature_Num_Level2+j]<=M2[j].minvalue) M2[j].minvalue=Feature2[(frame*frameSize2+temp2)*Feature_Num_Level2+j];
						}
					}
				}

			}
		}
	}
	//=========================================================================//
	double *level0_mean_positive = new double[Feature_Num_Level0]; memset(level0_mean_positive,0,sizeof(double)*Feature_Num_Level0);
	double *level1_mean_positive = new double[Feature_Num_Level1]; memset(level1_mean_positive,0,sizeof(double)*Feature_Num_Level1);
	double *level2_mean_positive = new double[Feature_Num_Level2]; memset(level2_mean_positive,0,sizeof(double)*Feature_Num_Level2);

	double *level0_mean_negative = new double[Feature_Num_Level0]; memset(level0_mean_negative,0,sizeof(double)*Feature_Num_Level0);
	double *level1_mean_negative = new double[Feature_Num_Level1]; memset(level1_mean_negative,0,sizeof(double)*Feature_Num_Level1);
	double *level2_mean_negative = new double[Feature_Num_Level2]; memset(level2_mean_negative,0,sizeof(double)*Feature_Num_Level2);

	int num_0_positive = 0, num_1_positive = 0, num_2_positive =0;
	int num_0_negative = 0, num_1_negative = 0, num_2_negative =0;

	for (int i=0;i<trainingnum;i++)
	{
		for (int j=0;j<frameSize0;j++)
		{
			if (+1 == Truth0[i*frameSize0+j]&&(j/W0<H00&&j%W0<W00))
			{
				for (int k=0;k<Feature_Num_Level0;k++)
				{
					double temp = Feature0[(i*frameSize0+j)*Feature_Num_Level0+k];
					level0_mean_positive[k] += -1 + 2.0*(temp - M0[k].minvalue)/(M0[k].maxvalue - M0[k].minvalue + 1e-8);
				}
				num_0_positive++;
			}
			else if (-1 == Truth0[i*frameSize0+j]&&(j/W0<H00&&j%W0<W00))
			{
				for (int k=0;k<Feature_Num_Level0;k++)
				{
					double temp = Feature0[(i*frameSize0+j)*Feature_Num_Level0+k];
					level0_mean_negative[k] += -1 + 2.0*(temp - M0[k].minvalue)/(M0[k].maxvalue - M0[k].minvalue + 1e-8);
				}
				num_0_negative++;
			}
		}
	}

	for (int frame=0;frame<trainingnum;frame++)
	{
		for (int i=0;i<frameSize0;i++)
		{
			int tempx=2*(i/W0),tempy=2*(i%W0);
			for (int l=0;l<4;l++)
			{
				int temp1=W1*(tempx+l/2)+(tempy+l%2);
				if(+1 == Truth1[4*i+l+frame*frameSize1]&&(temp1/W1<H11&&temp1%W1<W11)) 
				{
					for (int j=0;j<Feature_Num_Level1;j++)
					{
						double temp = Feature1[(frame*frameSize1+temp1)*Feature_Num_Level1+j];
						level1_mean_positive[j] += -1 + 2.0*(temp - M1[j].minvalue)/(M1[j].maxvalue - M1[j].minvalue + 1e-8);
					}
					num_1_positive++;
				}
				else if (-1 == Truth1[4*i+l+frame*frameSize1]&&(temp1/W1<H11&&temp1%W1<W11))
				{
					for (int j=0;j<Feature_Num_Level1;j++)
					{
						double temp = Feature1[(frame*frameSize1+temp1)*Feature_Num_Level1+j];
						level1_mean_negative[j] += -1 + 2.0*(temp - M1[j].minvalue)/(M1[j].maxvalue - M1[j].minvalue + 1e-8);
					}
					num_1_negative++;
				}
			}
		}
	}

	for (int frame=0;frame<trainingnum;frame++)
	{
		for (int i=0;i<frameSize0;i++)
		{
			int tempx=2*(i/W0),tempy=2*(i%W0);
			for (int l=0;l<4;l++)
			{
				int temp1=W1*(tempx+l/2)+(tempy+l%2);
				int tempxx=2*(temp1/W1), tempyy=2*(temp1%W1);
				for (int m=0;m<4;m++)
				{
					int temp2=W2*(tempxx+m/2)+(tempyy+m%2);
					if(+1 == Truth2[4*(4*i+l)+m+frame*frameSize2]&&(temp2/W2<H22&&temp2%W2<W22)) 
					{
						for (int j=0;j<Feature_Num_Level2;j++)
						{
							double temp = Feature2[(frame*frameSize2+temp2)*Feature_Num_Level2+j];
							level2_mean_positive[j] += -1 + 2.0*(temp - M2[j].minvalue)/(M2[j].maxvalue - M2[j].minvalue + 1e-8);
						}
						num_2_positive++;
					}
					else if (-1 == Truth2[4*(4*i+l)+m+frame*frameSize2]&&(temp2/W2<H22&&temp2%W2<W22))
					{
						for (int j=0;j<Feature_Num_Level2;j++)
						{
							double temp = Feature2[(frame*frameSize2+temp2)*Feature_Num_Level2+j];
							level2_mean_negative[j] += -1 + 2.0*(temp - M2[j].minvalue)/(M2[j].maxvalue - M2[j].minvalue + 1e-8);
						}
						num_2_negative++;
					}
				}
			}
		}
	}

	for (int k=0;k<Feature_Num_Level0;k++)
	{
		level0_mean_positive[k] /= 1.0*num_0_positive + 1e-8;
		level0_mean_negative[k] /= 1.0*num_0_negative + 1e-8;
	}
	for (int k=0;k<Feature_Num_Level1;k++)
	{
		level1_mean_positive[k] /= 1.0*num_1_positive + 1e-8;
		level1_mean_negative[k] /= 1.0*num_1_negative + 1e-8;
	}
	for (int k=0;k<Feature_Num_Level2;k++)
	{
		level2_mean_positive[k] /= 1.0*num_2_positive + 1e-8;
		level2_mean_negative[k] /= 1.0*num_2_negative + 1e-8;
	}
	//--------------------------------------------------------------------------
// 	for (int i=0;i<Feature_Num;i++)
// 	{
// 		cout<<M0[i].maxvalue<<" "<<M0[i].minvalue<<"|";
// 		cout<<M1[i].maxvalue<<" "<<M1[i].minvalue<<"|";
// 		cout<<M2[i].maxvalue<<" "<<M2[i].minvalue<<"|";
// 		cout<<M3[i].maxvalue<<" "<<M3[i].minvalue<<endl;
// 	}

	int count0, count1, count2;
	int negative_num0 = 0, negative_num1 = 0,negative_num2 = 0;
	int positive_num0 = 0, positive_num1 = 0, positive_num2 = 0;

	int negative_num00 = 0, negative_num11 = 0,negative_num22 = 0;
	int positive_num00 = 0, positive_num11 = 0, positive_num22 = 0;
	double weight_fuzzy = 0.00005;
	//1024*768/(64*64)=192
	int threshold_num0 = 192*80, threshold_num1 = 192*80, threshold_num2 = 192*80;
	
	for (int Class=0;Class<3;Class++)
	{
		int n=0,k=0;
		switch(Class)
		{
		case 0:
				for (int frame=0;frame<trainingnum;frame++)
				{
					for (int i=0;i<frameSize0;i++)
					{
						if(-1 == Truth0[frame*frameSize0+i]&&(i/W0<H00&&i%W0<W00))
						{
							negative_num0++;
						}
						else if(+1 == Truth0[frame*frameSize0+i]&&(i/W0<H00&&i%W0<W00))
						{
							positive_num0++;
						}
					}
				}
				count0 = (positive_num0+negative_num0)>=threshold_num0 ? threshold_num0 : positive_num0+negative_num0;
				prob0.l=count0; 
				prob0.x = Malloc(svm_node *,prob0.l);
				prob0.y = Malloc(double,prob0.l);
				prob0.weight = Malloc(double,prob0.l);
				data0 = Malloc(svm_node,prob0.l*(Feature_Num_Level0+1));
				//-----------------------------------------------------------------------------------------------------
				for (int frame=0;frame<trainingnum;frame++)
				{
					for (int i=0;i<frameSize0;i++)
					{			
						if (+1 == Truth0[frame*frameSize0+i]&&(i/W0<H00&&i%W0<W00))
						{
							prob0.y[n]=Truth0[frame*frameSize0+i];
							prob0.x[n]=&data0[k];
							double temp_weight0 = 0;
							f0<<prob0.y[n]<<" ";
							for (int j=0;j<Feature_Num_Level0;j++)
							{
								data0[k].index=j+1;
								f0<<j+1<<":";
								Double temp = Feature0[(frame*frameSize0+i)*Feature_Num_Level0+j];
								data0[k].value=-1+2*(temp-M0[j].minvalue)/(M0[j].maxvalue-M0[j].minvalue+1e-8);
								f0<<temp<<" ";
								temp_weight0 += (data0[k].value - level0_mean_positive[j])*(data0[k].value - level0_mean_positive[j]);
								k++;
							}
							f0<<endl;
#if Fuzzy_SVM
							temp_weight0 = temp_weight0;
#else
							temp_weight0 = 0;
#endif
							prob0.weight[n] = 2.0/(1.0 + exp(weight_fuzzy*temp_weight0));
							data0[k++].index=-1;
							n++;
							positive_num00++;
							if (n>=threshold_num0)
							{
								goto NEXT_00;
							}
						}
						else if (-1 == Truth0[frame*frameSize0+i]&&(i/W0<H00&&i%W0<W00))
						{
							prob0.y[n]=Truth0[frame*frameSize0+i];
							prob0.x[n]=&data0[k];
							double temp_weight0 = 0;
							f0<<prob0.y[n]<<" ";
							for (int j=0;j<Feature_Num_Level0;j++)
							{
								data0[k].index=j+1;
								f0<<j+1<<":";
								Double temp = Feature0[(frame*frameSize0+i)*Feature_Num_Level0+j];
								data0[k].value=-1+2*(temp-M0[j].minvalue)/(M0[j].maxvalue-M0[j].minvalue+1e-8);
								f0<<temp<<" ";
								temp_weight0 += (data0[k].value - level0_mean_negative[j])*(data0[k].value - level0_mean_negative[j]);
								k++;
							}
							f0<<endl;
#if Fuzzy_SVM
							temp_weight0 = temp_weight0;
#else
							temp_weight0 = 0;
#endif
							prob0.weight[n] = 2.0/(1.0 + exp(weight_fuzzy*temp_weight0));
							data0[k++].index=-1;
							n++;
							negative_num00++;
							if (n>=threshold_num0)
							{
								goto NEXT_00;
							}
						}
					}
				}
NEXT_00:
			 break;
			//-----------------------------------------------------------------------
		case 1:
			for (int frame=0;frame<trainingnum;frame++)
			{
				for (int i=0;i<frameSize0;i++)
				{
					int tempx=2*(i/W0),tempy=2*(i%W0);
					for (int l=0;l<4;l++)
					{
						int temp1=W1*(tempx+l/2)+(tempy+l%2);
						if(-1 ==Truth1[4*i+l+frame*frameSize1]&&(temp1/W1<H11&&temp1%W1<W11)) 
						{
							negative_num1++;
						}
						else if (+1 ==Truth1[4*i+l+frame*frameSize1]&&(temp1/W1<H11&&temp1%W1<W11))
						{
							positive_num1++;
						}
					}
				}
			}
			count1 = (positive_num1+negative_num1)>=threshold_num1 ? threshold_num1 : positive_num1+negative_num1;
			prob1.l=count1;
			prob1.x = Malloc(svm_node *,prob1.l);
			prob1.y = Malloc(double,prob1.l);
			prob1.weight = Malloc(double,prob1.l);
			data1 = Malloc(svm_node,prob1.l*(Feature_Num_Level1+1));
				//------------------------------------------------------------------------
			for (int frame=0;frame<trainingnum;frame++)
			{
				for (int i=0;i<frameSize0;i++)
				{
					int tempx=2*(i/W0),tempy=2*(i%W0);
					for (int l=0;l<4;l++)
					{
						int temp1=W1*(tempx+l/2)+(tempy+l%2);
						if(+1 ==Truth1[4*i+l+frame*frameSize1]&&(temp1/W1<H11&&temp1%W1<W11))
						{
							prob1.y[n]=Truth1[4*i+l+frame*frameSize1];
							prob1.x[n]=&data1[k];
							double temp_weight1 = 0;
							f1<<prob1.y[n]<<" ";
							for (int j=0;j<Feature_Num_Level1;j++)
							{
								data1[k].index=j+1;
								f1<<j+1<<":";
								double temp = Feature1[(frame*frameSize1+temp1)*Feature_Num_Level1+j];
								data1[k].value=-1+2*(temp-M1[j].minvalue)/(M1[j].maxvalue-M1[j].minvalue+1e-8);
								f1<<temp<<" ";
								temp_weight1 += (data1[k].value - level1_mean_positive[j])*(data1[k].value - level1_mean_positive[j]);
								k++;
							}
							f1<<endl;
#if Fuzzy_SVM
							temp_weight1 = temp_weight1;
#else
							temp_weight1 = 0;
#endif
							prob1.weight[n] = 2.0/(1.0 + exp(weight_fuzzy*temp_weight1));
							data1[k++].index=-1;
							n++;
							positive_num11++;
							if (n>=threshold_num1)
							{
								goto NEXT_10;
							}
						}
						else if (-1 ==Truth1[4*i+l+frame*frameSize1]&&(temp1/W1<H11&&temp1%W1<W11))
						{
							prob1.y[n]=Truth1[4*i+l+frame*frameSize1];
							prob1.x[n]=&data1[k];
							double temp_weight1 = 0;
							f1<<prob1.y[n]<<" ";
							for (int j=0;j<Feature_Num_Level1;j++)
							{
								data1[k].index=j+1;
								f1<<j+1<<":";
								double temp = Feature1[(frame*frameSize1+temp1)*Feature_Num_Level1+j];
								data1[k].value=-1+2*(temp-M1[j].minvalue)/(M1[j].maxvalue-M1[j].minvalue+1e-8);
								f1<<temp<<" ";
								temp_weight1 += (data1[k].value - level1_mean_negative[j])*(data1[k].value - level1_mean_negative[j]);
								k++;
							}
							f1<<endl;
#if Fuzzy_SVM
							temp_weight1 = temp_weight1;
#else
							temp_weight1 = 0;
#endif
							prob1.weight[n] = 2.0/(1.0 + exp(weight_fuzzy*temp_weight1));
							data1[k++].index=-1;
							n++;
							negative_num11++;
							if (n>=threshold_num1)
							{
								goto NEXT_10;
							}
						}
					}
				}
			}
NEXT_10:
			  break;
		case 2:
			for (int frame=0;frame<trainingnum;frame++)
			{
				for (int i=0;i<frameSize0;i++)
				{
					int tempx=2*(i/W0),tempy=2*(i%W0);
					for (int l=0;l<4;l++)
					{
						int temp1=W1*(tempx+l/2)+(tempy+l%2);
						int tempxx=2*(temp1/W1), tempyy=2*(temp1%W1);
						for (int m=0;m<4;m++)
						{
							int temp2=W2*(tempxx+m/2)+(tempyy+m%2);
							if(-1 ==Truth2[4*(4*i+l)+m+frame*frameSize2]&&(temp2/W2<H22&&temp2%W2<W22)) 
							{
								negative_num2++;
							}
							else if (+1 ==Truth2[4*(4*i+l)+m+frame*frameSize2]&&(temp2/W2<H22&&temp2%W2<W22))
							{
								positive_num2++;
							}
						}
					}
				}
			}
			count2 = (positive_num2+negative_num2)>=threshold_num2 ? threshold_num2 : positive_num2+negative_num2;
			prob2.l=count2;
			prob2.x = Malloc(svm_node *,prob2.l);
			prob2.y = Malloc(double,prob2.l);
			prob2.weight = Malloc(double,prob2.l);
			data2 = Malloc(svm_node,prob2.l*(Feature_Num_Level2+1));
			//-------------------------------------------------------------
			for (int frame=0;frame<trainingnum;frame++)
			{
				for (int i=0;i<frameSize0;i++)
				{
					int tempx=2*(i/W0),tempy=2*(i%W0);
					for (int l=0;l<4;l++)
					{
						int tempxx=2*tempx+2*(l/2),tempyy=2*tempy+2*(l%2);
						for (int m=0;m<4;m++)
						{
							int temp2=W2*(tempxx+m/2)+(tempyy+m%2);
							if(+1 ==Truth2[4*(4*i+l)+m+frame*frameSize2]&&(temp2/W2<H22&&temp2%W2<W22))
							{
								prob2.y[n]=Truth2[4*(4*i+l)+m+frame*frameSize2];
								prob2.x[n]=&data2[k];
								double temp_weight2 = 0;
								f2<<prob2.y[n]<<" ";
								for (int j=0;j<Feature_Num_Level2;j++)
								{
									data2[k].index=j+1;
									f2<<j+1<<":";
									double temp = Feature2[(frame*frameSize2+temp2)*Feature_Num_Level2+j];
									data2[k].value=-1+2*(temp-M2[j].minvalue)/(M2[j].maxvalue-M2[j].minvalue+1e-8);
									f2<<temp<<" ";
									temp_weight2 += (data2[k].value - level2_mean_positive[j])*(data2[k].value - level2_mean_positive[j]);
									k++;
								}
								f2<<endl;
#if Fuzzy_SVM
								temp_weight2 = temp_weight2;
#else
								temp_weight2 = 0;
#endif
								prob2.weight[n] = 2.0/(1.0 + exp(weight_fuzzy*temp_weight2));
								data2[k++].index=-1;
								n++;
								positive_num22++;;
								if(n>=threshold_num2)
								{
									goto NEXT_20;
								}
							}
							else if (-1 ==Truth2[4*(4*i+l)+m+frame*frameSize2]&&(temp2/W2<H22&&temp2%W2<W22))
							{
								prob2.y[n]=Truth2[4*(4*i+l)+m+frame*frameSize2];
								prob2.x[n]=&data2[k];
								double temp_weight2 = 0;
								f2<<prob2.y[n]<<" ";
								for (int j=0;j<Feature_Num_Level2;j++)
								{
									data2[k].index=j+1;
									f2<<j+1<<":";
									double temp = Feature2[(frame*frameSize2+temp2)*Feature_Num_Level2+j];
									data2[k].value=-1+2*(temp-M2[j].minvalue)/(M2[j].maxvalue-M2[j].minvalue+1e-8);
									f2<<temp<<" ";
									temp_weight2 += (data2[k].value - level2_mean_negative[j])*(data2[k].value - level2_mean_negative[j]);
									k++;
								}
								f2<<endl;
#if Fuzzy_SVM
								temp_weight2 = temp_weight2;
#else
								temp_weight2 = 0;
#endif
								prob2.weight[n] = 2.0/(1.0 + exp(weight_fuzzy*temp_weight2));
								data2[k++].index=-1;
								n++;
								negative_num22++;
								if(n>=threshold_num2)
								{
									goto NEXT_20;
								}
							}
						}
					}
				}
			}
NEXT_20:
			  break;
		 }
		//=====================================================================================================
		const char *error_msg=NULL;
		switch(Class)
		{
		case 0:
			//param.gamma=1.0/Feature_Num_Level0;
#if Different_Misclassification_Cost
			param.weight[0] = 1;
			param.weight[1] = 4.5;//zhulinwei
			
#else
			param.weight[0] = 1;
			param.weight[1] = 1;//zhulinwei
#endif
			if(negative_num00 > 2*positive_num00)
			{
				param.weight[0] = param.weight[0]*(2+1)/2;
				param.weight[1] = param.weight[1]*(2+1)/1;//zhulinwei
			}
			else if (positive_num00 > 2*negative_num0)
			{
				param.weight[0] = param.weight[0]*(2+1)/1;
				param.weight[1] = param.weight[1]*(2+1)/2;//zhulinwei
			}

			if (negative_num0 == 0 || positive_num0 == 0 )
			{
				param.weight[0] = 1.0;
				param.weight[1] = 1.0;//zhulinwei
			}

			error_msg=svm_check_parameter(&prob0,&param);
			if(error_msg)
			{
				fprintf(stderr,"ERROR: %s\n",error_msg);
				exit(1);
			}
			model0=svm_train(&prob0,&param);
			svm_save_model("model0.txt",model0);
			free(prob0.x);
			free(prob0.y);
			free(prob0.weight);
			//free(data0);
			break;
		case 1:
			//param.gamma=1.0/Feature_Num_Level1;
#if Different_Misclassification_Cost
			param.weight[0] = 1;
			param.weight[1] = 1.5;//zhulinwei
			
#else
			param.weight[0] = 1;
			param.weight[1] = 1;//zhulinwei
#endif
			if(negative_num11 > 2*positive_num11)
			{
				param.weight[0] = param.weight[0]*(2+1)/2;
				param.weight[1] = param.weight[1]*(2+1)/1;//zhulinwei
			}
			else if (positive_num11 > 2*negative_num11)
			{
				param.weight[0] = param.weight[0]*(2+1)/1;
				param.weight[1] = param.weight[1]*(2+1)/2;//zhulinwei
			}

			if (negative_num1 == 0 || positive_num1 == 0 )
			{
				param.weight[0] = 1.0;
				param.weight[1] = 1.0;//zhulinwei
			}

			error_msg=svm_check_parameter(&prob1,&param);
			if(error_msg)
			{
				fprintf(stderr,"ERROR: %s\n",error_msg);
				exit(1);
			}
			model1=svm_train(&prob1,&param);
			svm_save_model("model1.txt",model1);
			free(prob1.x);
			free(prob1.y);
			free(prob1.weight);
			//free(data1);
			break;
		case 2:
			//param.gamma=1.0/Feature_Num_Level2;
#if Different_Misclassification_Cost
			param.weight[0] = 1;
			param.weight[1] = 1.5;//zhulinwei
			
#else
			param.weight[0] = 1;
			param.weight[1] = 1;//zhulinwei
#endif
			if(negative_num22 > 2*positive_num22)
			{
				param.weight[0] = param.weight[0]*(2+1)/2;
				param.weight[1] = param.weight[1]*(2+1)/1;//zhulinwei
			}
			else if (positive_num22 > 2*negative_num22)
			{
				param.weight[0] = param.weight[0]*(2+1)/1;
				param.weight[1] = param.weight[1]*(2+1)/2;//zhulinwei
			}

			if (negative_num2 == 0 || positive_num2 == 0 )
			{
				param.weight[0] = 1.0;
				param.weight[1] = 1.0;//zhulinwei
			}

			error_msg=svm_check_parameter(&prob2,&param);
			if(error_msg)
			{
				fprintf(stderr,"ERROR: %s\n",error_msg);
				exit(1);
			}
			model2=svm_train(&prob2,&param);
			svm_save_model("model2.txt",model2);
			free(prob2.x);
			free(prob2.y);
			free(prob2.weight);
			//free(data2);
			break;
		}
     }
 }

Void TAppEncTop::xInitLibCfg()
{
  TComVPS vps;

  vps.setMaxTLayers                                               ( m_maxTempLayer );
  if (m_maxTempLayer == 1)
  {
    vps.setTemporalNestingFlag(true);
  }
  vps.setMaxLayers                                                ( 1 );
  for(Int i = 0; i < MAX_TLAYER; i++)
  {
    vps.setNumReorderPics                                         ( m_numReorderPics[i], i );
    vps.setMaxDecPicBuffering                                     ( m_maxDecPicBuffering[i], i );
  }
  m_cTEncTop.setVPS(&vps);

  m_cTEncTop.setProfile                                           ( m_profile);
  m_cTEncTop.setLevel                                             ( m_levelTier, m_level);
  m_cTEncTop.setProgressiveSourceFlag                             ( m_progressiveSourceFlag);
  m_cTEncTop.setInterlacedSourceFlag                              ( m_interlacedSourceFlag);
  m_cTEncTop.setNonPackedConstraintFlag                           ( m_nonPackedConstraintFlag);
  m_cTEncTop.setFrameOnlyConstraintFlag                           ( m_frameOnlyConstraintFlag);
  m_cTEncTop.setBitDepthConstraintValue                           ( m_bitDepthConstraint );
  m_cTEncTop.setChromaFormatConstraintValue                       ( m_chromaFormatConstraint );
  m_cTEncTop.setIntraConstraintFlag                               ( m_intraConstraintFlag );
  m_cTEncTop.setOnePictureOnlyConstraintFlag                      ( m_onePictureOnlyConstraintFlag );
  m_cTEncTop.setLowerBitRateConstraintFlag                        ( m_lowerBitRateConstraintFlag );

  m_cTEncTop.setPrintMSEBasedSequencePSNR                         ( m_printMSEBasedSequencePSNR);
  m_cTEncTop.setPrintFrameMSE                                     ( m_printFrameMSE);
  m_cTEncTop.setPrintSequenceMSE                                  ( m_printSequenceMSE);
  m_cTEncTop.setCabacZeroWordPaddingEnabled                       ( m_cabacZeroWordPaddingEnabled );

  m_cTEncTop.setFrameRate                                         ( m_iFrameRate );
  m_cTEncTop.setFrameSkip                                         ( m_FrameSkip );
  m_cTEncTop.setSourceWidth                                       ( m_iSourceWidth );
  m_cTEncTop.setSourceHeight                                      ( m_iSourceHeight );
  m_cTEncTop.setConformanceWindow                                 ( m_confWinLeft, m_confWinRight, m_confWinTop, m_confWinBottom );
  m_cTEncTop.setFramesToBeEncoded                                 ( m_framesToBeEncoded );

  m_cTEncTop.setLevel0(CU_OPTIMIZATION_LEVEL0);
  m_cTEncTop.setLevel1(CU_OPTIMIZATION_LEVEL1);
  m_cTEncTop.setLevel2(CU_OPTIMIZATION_LEVEL2);

  m_cTEncTop.setTP0(threshold_prob_0_positivie);
  m_cTEncTop.setTP1(threshold_prob_1_positivie);
  m_cTEncTop.setTP2(threshold_prob_2_positivie);

  m_cTEncTop.setTN0(threshold_prob_0_negative);
  m_cTEncTop.setTN1(threshold_prob_1_negative);
  m_cTEncTop.setTN2(threshold_prob_2_negative);//ZHULINWEI

  //====== Coding Structure ========
  m_cTEncTop.setIntraPeriod                                       ( m_iIntraPeriod );
  m_cTEncTop.setDecodingRefreshType                               ( m_iDecodingRefreshType );
  m_cTEncTop.setGOPSize                                           ( m_iGOPSize );
  m_cTEncTop.setGopList                                           ( m_GOPList );
  m_cTEncTop.setExtraRPSs                                         ( m_extraRPSs );
  for(Int i = 0; i < MAX_TLAYER; i++)
  {
    m_cTEncTop.setNumReorderPics                                  ( m_numReorderPics[i], i );
    m_cTEncTop.setMaxDecPicBuffering                              ( m_maxDecPicBuffering[i], i );
  }
  for( UInt uiLoop = 0; uiLoop < MAX_TLAYER; ++uiLoop )
  {
    m_cTEncTop.setLambdaModifier                                  ( uiLoop, m_adLambdaModifier[ uiLoop ] );
  }
  m_cTEncTop.setQP                                                ( m_iQP );

  m_cTEncTop.setPad                                               ( m_aiPad );

  m_cTEncTop.setMaxTempLayer                                      ( m_maxTempLayer );
  m_cTEncTop.setUseAMP( m_enableAMP );

  //===== Slice ========

  //====== Loop/Deblock Filter ========
  m_cTEncTop.setLoopFilterDisable                                 ( m_bLoopFilterDisable       );
  m_cTEncTop.setLoopFilterOffsetInPPS                             ( m_loopFilterOffsetInPPS );
  m_cTEncTop.setLoopFilterBetaOffset                              ( m_loopFilterBetaOffsetDiv2  );
  m_cTEncTop.setLoopFilterTcOffset                                ( m_loopFilterTcOffsetDiv2    );
  m_cTEncTop.setDeblockingFilterMetric                            ( m_DeblockingFilterMetric );

  //====== Motion search ========
  m_cTEncTop.setDisableIntraPUsInInterSlices                      ( m_bDisableIntraPUsInInterSlices );
  m_cTEncTop.setFastSearch                                        ( m_iFastSearch  );
  m_cTEncTop.setSearchRange                                       ( m_iSearchRange );
  m_cTEncTop.setBipredSearchRange                                 ( m_bipredSearchRange );
  m_cTEncTop.setClipForBiPredMeEnabled                            ( m_bClipForBiPredMeEnabled );
  m_cTEncTop.setFastMEAssumingSmootherMVEnabled                   ( m_bFastMEAssumingSmootherMVEnabled );

  //====== Quality control ========
  m_cTEncTop.setMaxDeltaQP                                        ( m_iMaxDeltaQP  );
  m_cTEncTop.setMaxCuDQPDepth                                     ( m_iMaxCuDQPDepth  );
  m_cTEncTop.setDiffCuChromaQpOffsetDepth                         ( m_diffCuChromaQpOffsetDepth );
  m_cTEncTop.setChromaCbQpOffset                                  ( m_cbQpOffset     );
  m_cTEncTop.setChromaCrQpOffset                                  ( m_crQpOffset  );

  m_cTEncTop.setChromaFormatIdc                                   ( m_chromaFormatIDC  );

#if ADAPTIVE_QP_SELECTION
  m_cTEncTop.setUseAdaptQpSelect                                  ( m_bUseAdaptQpSelect   );
#endif

  m_cTEncTop.setUseAdaptiveQP                                     ( m_bUseAdaptiveQP  );
  m_cTEncTop.setQPAdaptationRange                                 ( m_iQPAdaptationRange );
  m_cTEncTop.setExtendedPrecisionProcessingFlag                   ( m_extendedPrecisionProcessingFlag );
  m_cTEncTop.setHighPrecisionOffsetsEnabledFlag                   ( m_highPrecisionOffsetsEnabledFlag );
  //====== Tool list ========
  m_cTEncTop.setDeltaQpRD                                         ( m_uiDeltaQpRD  );
  m_cTEncTop.setUseASR                                            ( m_bUseASR      );
  m_cTEncTop.setUseHADME                                          ( m_bUseHADME    );
  m_cTEncTop.setdQPs                                              ( m_aidQP        );
  m_cTEncTop.setUseRDOQ                                           ( m_useRDOQ     );
  m_cTEncTop.setUseRDOQTS                                         ( m_useRDOQTS   );
#if T0196_SELECTIVE_RDOQ
  m_cTEncTop.setUseSelectiveRDOQ                                  ( m_useSelectiveRDOQ );
#endif
  m_cTEncTop.setRDpenalty                                         ( m_rdPenalty );
  m_cTEncTop.setMaxCUWidth                                        ( m_uiMaxCUWidth );
  m_cTEncTop.setMaxCUHeight                                       ( m_uiMaxCUHeight );
  m_cTEncTop.setMaxTotalCUDepth                                   ( m_uiMaxTotalCUDepth );
  m_cTEncTop.setLog2DiffMaxMinCodingBlockSize                     ( m_uiLog2DiffMaxMinCodingBlockSize );
  m_cTEncTop.setQuadtreeTULog2MaxSize                             ( m_uiQuadtreeTULog2MaxSize );
  m_cTEncTop.setQuadtreeTULog2MinSize                             ( m_uiQuadtreeTULog2MinSize );
  m_cTEncTop.setQuadtreeTUMaxDepthInter                           ( m_uiQuadtreeTUMaxDepthInter );
  m_cTEncTop.setQuadtreeTUMaxDepthIntra                           ( m_uiQuadtreeTUMaxDepthIntra );
  m_cTEncTop.setUseFastEnc                                        ( m_bUseFastEnc  );
  m_cTEncTop.setUseEarlyCU                                        ( m_bUseEarlyCU  );
  m_cTEncTop.setUseFastDecisionForMerge                           ( m_useFastDecisionForMerge  );
  m_cTEncTop.setUseCbfFastMode                                    ( m_bUseCbfFastMode  );
  m_cTEncTop.setUseEarlySkipDetection                             ( m_useEarlySkipDetection );
  m_cTEncTop.setCrossComponentPredictionEnabledFlag               ( m_crossComponentPredictionEnabledFlag );
  m_cTEncTop.setUseReconBasedCrossCPredictionEstimate             ( m_reconBasedCrossCPredictionEstimate );
  m_cTEncTop.setLog2SaoOffsetScale                                ( CHANNEL_TYPE_LUMA  , m_log2SaoOffsetScale[CHANNEL_TYPE_LUMA]   );
  m_cTEncTop.setLog2SaoOffsetScale                                ( CHANNEL_TYPE_CHROMA, m_log2SaoOffsetScale[CHANNEL_TYPE_CHROMA] );
  m_cTEncTop.setUseTransformSkip                                  ( m_useTransformSkip      );
  m_cTEncTop.setUseTransformSkipFast                              ( m_useTransformSkipFast  );
  m_cTEncTop.setTransformSkipRotationEnabledFlag                  ( m_transformSkipRotationEnabledFlag );
  m_cTEncTop.setTransformSkipContextEnabledFlag                   ( m_transformSkipContextEnabledFlag   );
  m_cTEncTop.setPersistentRiceAdaptationEnabledFlag               ( m_persistentRiceAdaptationEnabledFlag );
  m_cTEncTop.setCabacBypassAlignmentEnabledFlag                   ( m_cabacBypassAlignmentEnabledFlag );
  m_cTEncTop.setLog2MaxTransformSkipBlockSize                     ( m_log2MaxTransformSkipBlockSize  );
  for (UInt signallingModeIndex = 0; signallingModeIndex < NUMBER_OF_RDPCM_SIGNALLING_MODES; signallingModeIndex++)
  {
    m_cTEncTop.setRdpcmEnabledFlag                                ( RDPCMSignallingMode(signallingModeIndex), m_rdpcmEnabledFlag[signallingModeIndex]);
  }
  m_cTEncTop.setUseConstrainedIntraPred                           ( m_bUseConstrainedIntraPred );
  m_cTEncTop.setFastUDIUseMPMEnabled                              ( m_bFastUDIUseMPMEnabled );
  m_cTEncTop.setFastMEForGenBLowDelayEnabled                      ( m_bFastMEForGenBLowDelayEnabled );
  m_cTEncTop.setUseBLambdaForNonKeyLowDelayPictures               ( m_bUseBLambdaForNonKeyLowDelayPictures );
  m_cTEncTop.setPCMLog2MinSize                                    ( m_uiPCMLog2MinSize);
  m_cTEncTop.setUsePCM                                            ( m_usePCM );

  // set internal bit-depth and constants
  for (UInt channelType = 0; channelType < MAX_NUM_CHANNEL_TYPE; channelType++)
  {
    m_cTEncTop.setBitDepth((ChannelType)channelType, m_internalBitDepth[channelType]);
    m_cTEncTop.setPCMBitDepth((ChannelType)channelType, m_bPCMInputBitDepthFlag ? m_MSBExtendedBitDepth[channelType] : m_internalBitDepth[channelType]);
  }

  m_cTEncTop.setPCMLog2MaxSize                                    ( m_pcmLog2MaxSize);
  m_cTEncTop.setMaxNumMergeCand                                   ( m_maxNumMergeCand );


  //====== Weighted Prediction ========
  m_cTEncTop.setUseWP                                             ( m_useWeightedPred      );
  m_cTEncTop.setWPBiPred                                          ( m_useWeightedBiPred   );
  //====== Parallel Merge Estimation ========
  m_cTEncTop.setLog2ParallelMergeLevelMinus2                      ( m_log2ParallelMergeLevel - 2 );

  //====== Slice ========
  m_cTEncTop.setSliceMode                                         ( (SliceConstraint) m_sliceMode );
  m_cTEncTop.setSliceArgument                                     ( m_sliceArgument            );

  //====== Dependent Slice ========
  m_cTEncTop.setSliceSegmentMode                                  (  (SliceConstraint) m_sliceSegmentMode );
  m_cTEncTop.setSliceSegmentArgument                              ( m_sliceSegmentArgument     );

  if(m_sliceMode == NO_SLICES )
  {
    m_bLFCrossSliceBoundaryFlag = true;
  }
  m_cTEncTop.setLFCrossSliceBoundaryFlag                          ( m_bLFCrossSliceBoundaryFlag );
  m_cTEncTop.setUseSAO                                            ( m_bUseSAO );
  m_cTEncTop.setTestSAODisableAtPictureLevel                      ( m_bTestSAODisableAtPictureLevel );
  m_cTEncTop.setSaoEncodingRate                                   ( m_saoEncodingRate );
  m_cTEncTop.setSaoEncodingRateChroma                             ( m_saoEncodingRateChroma );
  m_cTEncTop.setMaxNumOffsetsPerPic                               ( m_maxNumOffsetsPerPic);

  m_cTEncTop.setSaoCtuBoundary                                    ( m_saoCtuBoundary);
  m_cTEncTop.setPCMInputBitDepthFlag                              ( m_bPCMInputBitDepthFlag);
  m_cTEncTop.setPCMFilterDisableFlag                              ( m_bPCMFilterDisableFlag);

  m_cTEncTop.setIntraSmoothingDisabledFlag                        (!m_enableIntraReferenceSmoothing );
  m_cTEncTop.setDecodedPictureHashSEIEnabled                      ( m_decodedPictureHashSEIEnabled );
  m_cTEncTop.setRecoveryPointSEIEnabled                           ( m_recoveryPointSEIEnabled );
  m_cTEncTop.setBufferingPeriodSEIEnabled                         ( m_bufferingPeriodSEIEnabled );
  m_cTEncTop.setPictureTimingSEIEnabled                           ( m_pictureTimingSEIEnabled );
  m_cTEncTop.setToneMappingInfoSEIEnabled                         ( m_toneMappingInfoSEIEnabled );
  m_cTEncTop.setTMISEIToneMapId                                   ( m_toneMapId );
  m_cTEncTop.setTMISEIToneMapCancelFlag                           ( m_toneMapCancelFlag );
  m_cTEncTop.setTMISEIToneMapPersistenceFlag                      ( m_toneMapPersistenceFlag );
  m_cTEncTop.setTMISEICodedDataBitDepth                           ( m_toneMapCodedDataBitDepth );
  m_cTEncTop.setTMISEITargetBitDepth                              ( m_toneMapTargetBitDepth );
  m_cTEncTop.setTMISEIModelID                                     ( m_toneMapModelId );
  m_cTEncTop.setTMISEIMinValue                                    ( m_toneMapMinValue );
  m_cTEncTop.setTMISEIMaxValue                                    ( m_toneMapMaxValue );
  m_cTEncTop.setTMISEISigmoidMidpoint                             ( m_sigmoidMidpoint );
  m_cTEncTop.setTMISEISigmoidWidth                                ( m_sigmoidWidth );
  m_cTEncTop.setTMISEIStartOfCodedInterva                         ( m_startOfCodedInterval );
  m_cTEncTop.setTMISEINumPivots                                   ( m_numPivots );
  m_cTEncTop.setTMISEICodedPivotValue                             ( m_codedPivotValue );
  m_cTEncTop.setTMISEITargetPivotValue                            ( m_targetPivotValue );
  m_cTEncTop.setTMISEICameraIsoSpeedIdc                           ( m_cameraIsoSpeedIdc );
  m_cTEncTop.setTMISEICameraIsoSpeedValue                         ( m_cameraIsoSpeedValue );
  m_cTEncTop.setTMISEIExposureIndexIdc                            ( m_exposureIndexIdc );
  m_cTEncTop.setTMISEIExposureIndexValue                          ( m_exposureIndexValue );
  m_cTEncTop.setTMISEIExposureCompensationValueSignFlag           ( m_exposureCompensationValueSignFlag );
  m_cTEncTop.setTMISEIExposureCompensationValueNumerator          ( m_exposureCompensationValueNumerator );
  m_cTEncTop.setTMISEIExposureCompensationValueDenomIdc           ( m_exposureCompensationValueDenomIdc );
  m_cTEncTop.setTMISEIRefScreenLuminanceWhite                     ( m_refScreenLuminanceWhite );
  m_cTEncTop.setTMISEIExtendedRangeWhiteLevel                     ( m_extendedRangeWhiteLevel );
  m_cTEncTop.setTMISEINominalBlackLevelLumaCodeValue              ( m_nominalBlackLevelLumaCodeValue );
  m_cTEncTop.setTMISEINominalWhiteLevelLumaCodeValue              ( m_nominalWhiteLevelLumaCodeValue );
  m_cTEncTop.setTMISEIExtendedWhiteLevelLumaCodeValue             ( m_extendedWhiteLevelLumaCodeValue );
  m_cTEncTop.setChromaSamplingFilterHintEnabled                   ( m_chromaSamplingFilterSEIenabled );
  m_cTEncTop.setChromaSamplingHorFilterIdc                        ( m_chromaSamplingHorFilterIdc );
  m_cTEncTop.setChromaSamplingVerFilterIdc                        ( m_chromaSamplingVerFilterIdc );
  m_cTEncTop.setFramePackingArrangementSEIEnabled                 ( m_framePackingSEIEnabled );
  m_cTEncTop.setFramePackingArrangementSEIType                    ( m_framePackingSEIType );
  m_cTEncTop.setFramePackingArrangementSEIId                      ( m_framePackingSEIId );
  m_cTEncTop.setFramePackingArrangementSEIQuincunx                ( m_framePackingSEIQuincunx );
  m_cTEncTop.setFramePackingArrangementSEIInterpretation          ( m_framePackingSEIInterpretation );
  m_cTEncTop.setSegmentedRectFramePackingArrangementSEIEnabled    ( m_segmentedRectFramePackingSEIEnabled );
  m_cTEncTop.setSegmentedRectFramePackingArrangementSEICancel     ( m_segmentedRectFramePackingSEICancel );
  m_cTEncTop.setSegmentedRectFramePackingArrangementSEIType       ( m_segmentedRectFramePackingSEIType );
  m_cTEncTop.setSegmentedRectFramePackingArrangementSEIPersistence( m_segmentedRectFramePackingSEIPersistence );
  m_cTEncTop.setDisplayOrientationSEIAngle                        ( m_displayOrientationSEIAngle );
  m_cTEncTop.setTemporalLevel0IndexSEIEnabled                     ( m_temporalLevel0IndexSEIEnabled );
  m_cTEncTop.setGradualDecodingRefreshInfoEnabled                 ( m_gradualDecodingRefreshInfoEnabled );
  m_cTEncTop.setNoDisplaySEITLayer                                ( m_noDisplaySEITLayer );
  m_cTEncTop.setDecodingUnitInfoSEIEnabled                        ( m_decodingUnitInfoSEIEnabled );
  m_cTEncTop.setSOPDescriptionSEIEnabled                          ( m_SOPDescriptionSEIEnabled );
  m_cTEncTop.setScalableNestingSEIEnabled                         ( m_scalableNestingSEIEnabled );
  m_cTEncTop.setTMCTSSEIEnabled                                   ( m_tmctsSEIEnabled );
  m_cTEncTop.setTimeCodeSEIEnabled                                ( m_timeCodeSEIEnabled );
  m_cTEncTop.setNumberOfTimeSets                                  ( m_timeCodeSEINumTs );
  for(Int i = 0; i < m_timeCodeSEINumTs; i++)
  {
    m_cTEncTop.setTimeSet(m_timeSetArray[i], i);
  }
  m_cTEncTop.setKneeSEIEnabled                                    ( m_kneeSEIEnabled );
  m_cTEncTop.setKneeSEIId                                         ( m_kneeSEIId );
  m_cTEncTop.setKneeSEICancelFlag                                 ( m_kneeSEICancelFlag );
  m_cTEncTop.setKneeSEIPersistenceFlag                            ( m_kneeSEIPersistenceFlag );
  m_cTEncTop.setKneeSEIInputDrange                                ( m_kneeSEIInputDrange );
  m_cTEncTop.setKneeSEIInputDispLuminance                         ( m_kneeSEIInputDispLuminance );
  m_cTEncTop.setKneeSEIOutputDrange                               ( m_kneeSEIOutputDrange );
  m_cTEncTop.setKneeSEIOutputDispLuminance                        ( m_kneeSEIOutputDispLuminance );
  m_cTEncTop.setKneeSEINumKneePointsMinus1                        ( m_kneeSEINumKneePointsMinus1 );
  m_cTEncTop.setKneeSEIInputKneePoint                             ( m_kneeSEIInputKneePoint );
  m_cTEncTop.setKneeSEIOutputKneePoint                            ( m_kneeSEIOutputKneePoint );
  m_cTEncTop.setMasteringDisplaySEI                               ( m_masteringDisplay );

  m_cTEncTop.setTileUniformSpacingFlag                            ( m_tileUniformSpacingFlag );
  m_cTEncTop.setNumColumnsMinus1                                  ( m_numTileColumnsMinus1 );
  m_cTEncTop.setNumRowsMinus1                                     ( m_numTileRowsMinus1 );
  if(!m_tileUniformSpacingFlag)
  {
    m_cTEncTop.setColumnWidth                                     ( m_tileColumnWidth );
    m_cTEncTop.setRowHeight                                       ( m_tileRowHeight );
  }
  m_cTEncTop.xCheckGSParameters();
  Int uiTilesCount = (m_numTileRowsMinus1+1) * (m_numTileColumnsMinus1+1);
  if(uiTilesCount == 1)
  {
    m_bLFCrossTileBoundaryFlag = true;
  }
  m_cTEncTop.setLFCrossTileBoundaryFlag                           ( m_bLFCrossTileBoundaryFlag );
  m_cTEncTop.setWaveFrontSynchro                                  ( m_iWaveFrontSynchro );
  m_cTEncTop.setTMVPModeId                                        ( m_TMVPModeId );
  m_cTEncTop.setUseScalingListId                                  ( m_useScalingListId  );
  m_cTEncTop.setScalingListFile                                   ( m_scalingListFile   );
  m_cTEncTop.setSaliencyFile                                      ( m_pchSaliencyFile   );//zhulinwei
  m_cTEncTop.setSignHideFlag                                      ( m_signHideFlag);
  m_cTEncTop.setUseRateCtrl                                       ( m_RCEnableRateControl );
  m_cTEncTop.setTargetBitrate                                     ( m_RCTargetBitrate );
  m_cTEncTop.setKeepHierBit                                       ( m_RCKeepHierarchicalBit );
  m_cTEncTop.setLCULevelRC                                        ( m_RCLCULevelRC );
  m_cTEncTop.setUseLCUSeparateModel                               ( m_RCUseLCUSeparateModel );
  m_cTEncTop.setInitialQP                                         ( m_RCInitialQP );
  m_cTEncTop.setForceIntraQP                                      ( m_RCForceIntraQP );
  m_cTEncTop.setTransquantBypassEnableFlag                        ( m_TransquantBypassEnableFlag );
  m_cTEncTop.setCUTransquantBypassFlagForceValue                  ( m_CUTransquantBypassFlagForce );
  m_cTEncTop.setCostMode                                          ( m_costMode );
  m_cTEncTop.setUseRecalculateQPAccordingToLambda                 ( m_recalculateQPAccordingToLambda );
  m_cTEncTop.setUseStrongIntraSmoothing                           ( m_useStrongIntraSmoothing );
  m_cTEncTop.setActiveParameterSetsSEIEnabled                     ( m_activeParameterSetsSEIEnabled );
  m_cTEncTop.setVuiParametersPresentFlag                          ( m_vuiParametersPresentFlag );
  m_cTEncTop.setAspectRatioInfoPresentFlag                        ( m_aspectRatioInfoPresentFlag);
  m_cTEncTop.setAspectRatioIdc                                    ( m_aspectRatioIdc );
  m_cTEncTop.setSarWidth                                          ( m_sarWidth );
  m_cTEncTop.setSarHeight                                         ( m_sarHeight );
  m_cTEncTop.setOverscanInfoPresentFlag                           ( m_overscanInfoPresentFlag );
  m_cTEncTop.setOverscanAppropriateFlag                           ( m_overscanAppropriateFlag );
  m_cTEncTop.setVideoSignalTypePresentFlag                        ( m_videoSignalTypePresentFlag );
  m_cTEncTop.setVideoFormat                                       ( m_videoFormat );
  m_cTEncTop.setVideoFullRangeFlag                                ( m_videoFullRangeFlag );
  m_cTEncTop.setColourDescriptionPresentFlag                      ( m_colourDescriptionPresentFlag );
  m_cTEncTop.setColourPrimaries                                   ( m_colourPrimaries );
  m_cTEncTop.setTransferCharacteristics                           ( m_transferCharacteristics );
  m_cTEncTop.setMatrixCoefficients                                ( m_matrixCoefficients );
  m_cTEncTop.setChromaLocInfoPresentFlag                          ( m_chromaLocInfoPresentFlag );
  m_cTEncTop.setChromaSampleLocTypeTopField                       ( m_chromaSampleLocTypeTopField );
  m_cTEncTop.setChromaSampleLocTypeBottomField                    ( m_chromaSampleLocTypeBottomField );
  m_cTEncTop.setNeutralChromaIndicationFlag                       ( m_neutralChromaIndicationFlag );
  m_cTEncTop.setDefaultDisplayWindow                              ( m_defDispWinLeftOffset, m_defDispWinRightOffset, m_defDispWinTopOffset, m_defDispWinBottomOffset );
  m_cTEncTop.setFrameFieldInfoPresentFlag                         ( m_frameFieldInfoPresentFlag );
  m_cTEncTop.setPocProportionalToTimingFlag                       ( m_pocProportionalToTimingFlag );
  m_cTEncTop.setNumTicksPocDiffOneMinus1                          ( m_numTicksPocDiffOneMinus1    );
  m_cTEncTop.setBitstreamRestrictionFlag                          ( m_bitstreamRestrictionFlag );
  m_cTEncTop.setTilesFixedStructureFlag                           ( m_tilesFixedStructureFlag );
  m_cTEncTop.setMotionVectorsOverPicBoundariesFlag                ( m_motionVectorsOverPicBoundariesFlag );
  m_cTEncTop.setMinSpatialSegmentationIdc                         ( m_minSpatialSegmentationIdc );
  m_cTEncTop.setMaxBytesPerPicDenom                               ( m_maxBytesPerPicDenom );
  m_cTEncTop.setMaxBitsPerMinCuDenom                              ( m_maxBitsPerMinCuDenom );
  m_cTEncTop.setLog2MaxMvLengthHorizontal                         ( m_log2MaxMvLengthHorizontal );
  m_cTEncTop.setLog2MaxMvLengthVertical                           ( m_log2MaxMvLengthVertical );
  m_cTEncTop.setEfficientFieldIRAPEnabled                         ( m_bEfficientFieldIRAPEnabled );
  m_cTEncTop.setHarmonizeGopFirstFieldCoupleEnabled               ( m_bHarmonizeGopFirstFieldCoupleEnabled );

  m_cTEncTop.setSummaryOutFilename                                ( m_summaryOutFilename );
  m_cTEncTop.setSummaryPicFilenameBase                            ( m_summaryPicFilenameBase );
  m_cTEncTop.setSummaryVerboseness                                ( m_summaryVerboseness );
}

Void TAppEncTop::xCreateLib()
{
  // Video I/O
  m_cTVideoIOYuvInputFile.open( m_pchInputFile,     false, m_inputBitDepth, m_MSBExtendedBitDepth, m_internalBitDepth );  // read  mode
  m_cTVideoIOYuvInputFile.skipFrames(m_FrameSkip, m_iSourceWidth - m_aiPad[0], m_iSourceHeight - m_aiPad[1], m_InputChromaFormatIDC);

  if (m_pchReconFile)
  {
    m_cTVideoIOYuvReconFile.open(m_pchReconFile, true, m_outputBitDepth, m_outputBitDepth, m_internalBitDepth);  // write mode
  }

  // Neo Decoder
  m_cTEncTop.create();
}

Void TAppEncTop::xDestroyLib()
{
  // Video I/O
  m_cTVideoIOYuvInputFile.close();
  m_cTVideoIOYuvReconFile.close();

  // Neo Decoder
  m_cTEncTop.destroy();
}

Void TAppEncTop::xInitLib(Bool isFieldCoding)
{
  m_cTEncTop.init(isFieldCoding);
}

// ====================================================================================================================
// Public member functions
// ====================================================================================================================

/**
 - create internal class
 - initialize internal variable
 - until the end of input YUV file, call encoding function in TEncTop class
 - delete allocated buffers
 - destroy internal class
 .
 */
Void TAppEncTop::encode()
{
  fstream bitstreamFile(m_pchBitstreamFile, fstream::binary | fstream::out);
  if (!bitstreamFile)
  {
    fprintf(stderr, "\nfailed to open bitstream file `%s' for writing\n", m_pchBitstreamFile);
    exit(EXIT_FAILURE);
  }

  TComPicYuv*       pcPicYuvOrg = new TComPicYuv;
  TComPicYuv*       pcPicYuvRec = NULL;

  // initialize internal class & member variables
  xInitLibCfg();
  xCreateLib();
  xInitLib(m_isField);

  printChromaFormat();

  // main encoder loop
  Int   iNumEncoded = 0;
  Bool  bEos = false;

  const InputColourSpaceConversion ipCSC  =  m_inputColourSpaceConvert;
  const InputColourSpaceConversion snrCSC = (!m_snrInternalColourSpace) ? m_inputColourSpaceConvert : IPCOLOURSPACE_UNCHANGED;

  list<AccessUnit> outputAccessUnits; ///< list of access units to write out.  is populated by the encoding process

  TComPicYuv cPicYuvTrueOrg;

  // allocate original YUV buffer
  if( m_isField )
  {
    pcPicYuvOrg->create  ( m_iSourceWidth, m_iSourceHeightOrg, m_chromaFormatIDC, m_uiMaxCUWidth, m_uiMaxCUHeight, m_uiMaxTotalCUDepth, true );
    cPicYuvTrueOrg.create(m_iSourceWidth, m_iSourceHeightOrg, m_chromaFormatIDC, m_uiMaxCUWidth, m_uiMaxCUHeight, m_uiMaxTotalCUDepth, true);
  }
  else
  {
    pcPicYuvOrg->create  ( m_iSourceWidth, m_iSourceHeight, m_chromaFormatIDC, m_uiMaxCUWidth, m_uiMaxCUHeight, m_uiMaxTotalCUDepth, true );
    cPicYuvTrueOrg.create(m_iSourceWidth, m_iSourceHeight, m_chromaFormatIDC, m_uiMaxCUWidth, m_uiMaxCUHeight, m_uiMaxTotalCUDepth, true );
  }

#if Machine_Learning_Debug //ZHULINWEI

  Int WIDTH  = m_iSourceWidth;
  Int HEIGHT = m_iSourceHeight;

#if FULL_FEATURE
  Int Feature_Num_Level0 = 13, Feature_Num_Level1 = 13, Feature_Num_Level2 = 13;//the number of features
#endif
#if SELECTED_FEATURE
  Int Feature_Num_Level0 = 7, Feature_Num_Level1 = 4, Feature_Num_Level2 = 12;//the number of features
#endif
  //Int trainGOP = 1;
  Int frameSize0,frameSize1,frameSize2;
  //--------------------------边界考虑---------------------------------//
  if (WIDTH%64==0&&HEIGHT%64==0) 
	  frameSize0 = WIDTH/64*HEIGHT/64;
  else if (WIDTH%64!=0&&HEIGHT%64==0)
	  frameSize0 = (WIDTH/64+1)*HEIGHT/64;
  else if (WIDTH%64==0&&HEIGHT%64!=0)
	  frameSize0 = WIDTH/64*(HEIGHT/64+1);
  else if (WIDTH%64!=0&&HEIGHT%64!=0)
	  frameSize0 = (WIDTH/64+1)*(HEIGHT/64+1);
  //------------------------------------------------------------------//
  frameSize1=4*frameSize0;frameSize2=16*frameSize0;
  Int W0,W1,W2,H0,H1,H2;

  if (WIDTH%64==0)  W0 = WIDTH/64;
  else              W0 = WIDTH/64+1;

  W1=2*W0;W2=4*W0;

  if (HEIGHT%64==0) H0 = HEIGHT/64;
  else              H0 = HEIGHT/64+1;

  H1=2*H0;H2=4*H0;

#if Saliency_Weight
  Mat img(Size(WIDTH,HEIGHT),CV_8UC3);
  Mat img_output(Size(WIDTH,HEIGHT),CV_8UC3);
#endif
  //------------------------------------------------------------------//
  Int trainframe = trainGOP*m_iGOPSize;
  //------------------------------------------------------------------//
  Double *Feature0 = new Double[trainframe*frameSize0*Feature_Num_Level0];
  Double *Feature1 = new Double[trainframe*frameSize1*Feature_Num_Level1];
  Double *Feature2 = new Double[trainframe*frameSize2*Feature_Num_Level2];
  memset(Feature0,0,sizeof(Double)*trainframe*frameSize0*Feature_Num_Level0);
  memset(Feature1,0,sizeof(Double)*trainframe*frameSize1*Feature_Num_Level1);
  memset(Feature2,0,sizeof(Double)*trainframe*frameSize2*Feature_Num_Level2);


  Double *Feature00 = new Double[trainframe*frameSize0*Feature_Num_Level0];
  Double *Feature11 = new Double[trainframe*frameSize1*Feature_Num_Level1];
  Double *Feature22 = new Double[trainframe*frameSize2*Feature_Num_Level2];
  memset(Feature00,0,sizeof(Double)*trainframe*frameSize0*Feature_Num_Level0);
  memset(Feature11,0,sizeof(Double)*trainframe*frameSize1*Feature_Num_Level1);
  memset(Feature22,0,sizeof(Double)*trainframe*frameSize2*Feature_Num_Level2);


  Int *Truth0 = new Int[trainframe*frameSize0];
  Int *Truth1 = new Int[trainframe*frameSize1];
  Int *Truth2 = new Int[trainframe*frameSize2];

  maxmin *M0,*M1,*M2;
  M0=new maxmin [Feature_Num_Level0];
  M1=new maxmin [Feature_Num_Level1];
  M2=new maxmin [Feature_Num_Level2];
#if Online
  svm_model *model0 = NULL, *model1 = NULL, *model2 = NULL;
#endif
  bool training = false;
  //------------------------------------------------------------------//
#if Offline
  svm_model *model0 = svm_load_model("model0.txt");
  svm_model *model1 = svm_load_model("model1.txt");
  svm_model *model2 = svm_load_model("model2.txt");
  svm_model *model3 = svm_load_model("model3.txt");

  FILE *paramater0 = fopen("par0.txt","rb");
  FILE *paramater1 = fopen("par1.txt","rb");
  FILE *paramater2 = fopen("par2.txt","rb");
 
  char string_temp;int classifier[2];int index0;
//==============================================================================
  fseek(paramater0,0,0);
  fscanf(paramater0,"%c",&string_temp);
  fscanf(paramater0,"%d %d",&classifier[0],&classifier[1]);
  for (int k=0;k<Feature_Num;k++)
  {
	  fscanf(paramater0,"%d %lf %lf",&index0,&M0[k].minvalue,&M0[k].maxvalue);
  }
//==============================================================================
  fseek(paramater1,0,0);
  fscanf(paramater1,"%c",&string_temp);
  fscanf(paramater1,"%d %d",&classifier[0],&classifier[1]);
  for (int k=0;k<Feature_Num;k++)
  {
	  fscanf(paramater1,"%d %lf %lf",&index0,&M1[k].minvalue,&M1[k].maxvalue);
  }
//==============================================================================
  fseek(paramater2,0,0);
  fscanf(paramater2,"%c",&string_temp);
  fscanf(paramater2,"%d %d",&classifier[0],&classifier[1]);
  for (int k=0;k<Feature_Num;k++)
  {
	  fscanf(paramater2,"%d %lf %lf",&index0,&M2[k].minvalue,&M2[k].maxvalue);
  }
#endif

#endif

  while ( !bEos )
  {
    // get buffers
    xGetBuffer(pcPicYuvRec);

    // read input YUV file
    m_cTVideoIOYuvInputFile.read( pcPicYuvOrg, &cPicYuvTrueOrg, ipCSC, m_aiPad, m_InputChromaFormatIDC, m_bClipInputVideoToRec709Range );

    // increase number of received frames
    m_iFrameRcvd++;

    bEos = (m_isField && (m_iFrameRcvd == (m_framesToBeEncoded >> 1) )) || ( !m_isField && (m_iFrameRcvd == m_framesToBeEncoded) );

    Bool flush = 0;
    // if end of file (which is only detected on a read failure) flush the encoder of any queued pictures
    if (m_cTVideoIOYuvInputFile.isEof())
    {
      flush = true;
      bEos = true;
      m_iFrameRcvd--;
      m_cTEncTop.setFramesToBeEncoded(m_iFrameRcvd);
    }

    // call encoding function for one frame
#if Machine_Learning_Debug
#if Online
	if (training == true)
	{
		cout<<" classifier generating........................................."<<endl;
#if Online
		SVM_Train_Online(Feature0,Feature1,Feature2,Truth0,Truth1,Truth2,M0,M1,M2,
			Feature_Num_Level0, Feature_Num_Level1, Feature_Num_Level2,trainframe,frameSize0,frameSize1,frameSize2,W0,W1,W2,
			WIDTH,HEIGHT,model0,model1,model2);
#endif
		training = false;

		cout<<" classifier finished............................................"<<endl;
	}
#endif

	if (m_iFrameRcvd == 1)//Intra
	{
		if ( m_isField )
		{
			m_cTEncTop.encode( bEos, flush ? 0 : pcPicYuvOrg, flush ? 0 : &cPicYuvTrueOrg, snrCSC, m_cListPicYuvRec, outputAccessUnits, iNumEncoded, m_isTopFieldFirst );
		}
		else
		{
			m_cTEncTop.encode( bEos, flush ? 0 : pcPicYuvOrg, flush ? 0 : &cPicYuvTrueOrg, snrCSC, m_cListPicYuvRec, outputAccessUnits, iNumEncoded );
		}
	}
#if Online
	else if(((m_iFrameRcvd -2)/4)%25 == 0)//train   适用于LDP,对于RA,需要对应调整， 周期性更新模型，每隔100frames
	{
		if ( m_isField )
		{
			m_cTEncTop.encode_train( Feature0,Feature1,Feature2,
				Truth0,Truth1,Truth2,
				frameSize0,frameSize1,frameSize2,Feature_Num_Level0, Feature_Num_Level1, Feature_Num_Level2,
				bEos, flush ? 0 : pcPicYuvOrg, flush ? 0 : &cPicYuvTrueOrg, snrCSC, m_cListPicYuvRec, outputAccessUnits, iNumEncoded, m_isTopFieldFirst );
		}
		else
		{
			m_cTEncTop.encode_train( Feature0,Feature1,Feature2,
				Truth0,Truth1,Truth2,
				frameSize0,frameSize1,frameSize2,Feature_Num_Level0, Feature_Num_Level1, Feature_Num_Level2,
				bEos, flush ? 0 : pcPicYuvOrg, flush ? 0 : &cPicYuvTrueOrg, snrCSC, m_cListPicYuvRec, outputAccessUnits, iNumEncoded );
		}
		if ((m_iFrameRcvd -1)%4 == 0) //适用于LDP,RA,需要对应调整
		{
			training = true;
		}
	}
#endif
	else //predict
	{
#if Online || Offline
		if ( m_isField )
		{
			m_cTEncTop.encode_predict_online( Feature00,Feature11,Feature22,
				model0,model1,model2,M0,M1,M2,
				frameSize0,frameSize1,frameSize2,Feature_Num_Level0, Feature_Num_Level1, Feature_Num_Level2,
#if Saliency_Weight
				img,img_output,
#endif
				bEos, flush ? 0 : pcPicYuvOrg, flush ? 0 : &cPicYuvTrueOrg, snrCSC, m_cListPicYuvRec, outputAccessUnits, iNumEncoded, m_isTopFieldFirst );
		}
		else
		{
			m_cTEncTop.encode_predict_online( Feature00,Feature11,Feature22,
				model0,model1,model2,M0,M1,M2,
				frameSize0,frameSize1,frameSize2,Feature_Num_Level0, Feature_Num_Level1, Feature_Num_Level2,
#if Saliency_Weight
				img,img_output,
#endif
				bEos, flush ? 0 : pcPicYuvOrg, flush ? 0 : &cPicYuvTrueOrg, snrCSC, m_cListPicYuvRec, outputAccessUnits, iNumEncoded );
		}
#endif
	}

#else
	if ( m_isField )
	{
		m_cTEncTop.encode( bEos, flush ? 0 : pcPicYuvOrg, flush ? 0 : &cPicYuvTrueOrg, snrCSC, m_cListPicYuvRec, outputAccessUnits, iNumEncoded, m_isTopFieldFirst );
	}
	else
	{
		m_cTEncTop.encode( bEos, flush ? 0 : pcPicYuvOrg, flush ? 0 : &cPicYuvTrueOrg, snrCSC, m_cListPicYuvRec, outputAccessUnits, iNumEncoded );
	}
#endif
   

    // write bistream to file if necessary
    if ( iNumEncoded > 0 )
    {
      xWriteOutput(bitstreamFile, iNumEncoded, outputAccessUnits);
      outputAccessUnits.clear();
    }
  }

  m_cTEncTop.printSummary(m_isField);

  // delete original YUV buffer
  pcPicYuvOrg->destroy();
  delete pcPicYuvOrg;
  pcPicYuvOrg = NULL;

  // delete used buffers in encoder class
  m_cTEncTop.deletePicBuffer();
  cPicYuvTrueOrg.destroy();

  // delete buffers & classes
  xDeleteBuffer();
  xDestroyLib();

  printRateSummary();

  return;
}

// ====================================================================================================================
// Protected member functions
// ====================================================================================================================

/**
 - application has picture buffer list with size of GOP
 - picture buffer list acts as ring buffer
 - end of the list has the latest picture
 .
 */
Void TAppEncTop::xGetBuffer( TComPicYuv*& rpcPicYuvRec)
{
  assert( m_iGOPSize > 0 );

  // org. buffer
  if ( m_cListPicYuvRec.size() >= (UInt)m_iGOPSize ) // buffer will be 1 element longer when using field coding, to maintain first field whilst processing second.
  {
    rpcPicYuvRec = m_cListPicYuvRec.popFront();

  }
  else
  {
    rpcPicYuvRec = new TComPicYuv;

    rpcPicYuvRec->create( m_iSourceWidth, m_iSourceHeight, m_chromaFormatIDC, m_uiMaxCUWidth, m_uiMaxCUHeight, m_uiMaxTotalCUDepth, true );

  }
  m_cListPicYuvRec.pushBack( rpcPicYuvRec );
}

Void TAppEncTop::xDeleteBuffer( )
{
  TComList<TComPicYuv*>::iterator iterPicYuvRec  = m_cListPicYuvRec.begin();

  Int iSize = Int( m_cListPicYuvRec.size() );

  for ( Int i = 0; i < iSize; i++ )
  {
    TComPicYuv*  pcPicYuvRec  = *(iterPicYuvRec++);
    pcPicYuvRec->destroy();
    delete pcPicYuvRec; pcPicYuvRec = NULL;
  }

}

/** 
  Write access units to output file.
  \param bitstreamFile  target bitstream file
  \param iNumEncoded    number of encoded frames
  \param accessUnits    list of access units to be written
 */
Void TAppEncTop::xWriteOutput(std::ostream& bitstreamFile, Int iNumEncoded, const std::list<AccessUnit>& accessUnits)
{
  const InputColourSpaceConversion ipCSC = (!m_outputInternalColourSpace) ? m_inputColourSpaceConvert : IPCOLOURSPACE_UNCHANGED;

  if (m_isField)
  {
    //Reinterlace fields
    Int i;
    TComList<TComPicYuv*>::iterator iterPicYuvRec = m_cListPicYuvRec.end();
    list<AccessUnit>::const_iterator iterBitstream = accessUnits.begin();

    for ( i = 0; i < iNumEncoded; i++ )
    {
      --iterPicYuvRec;
    }

    for ( i = 0; i < iNumEncoded/2; i++ )
    {
      TComPicYuv*  pcPicYuvRecTop  = *(iterPicYuvRec++);
      TComPicYuv*  pcPicYuvRecBottom  = *(iterPicYuvRec++);

      if (m_pchReconFile)
      {
        m_cTVideoIOYuvReconFile.write( pcPicYuvRecTop, pcPicYuvRecBottom, ipCSC, m_confWinLeft, m_confWinRight, m_confWinTop, m_confWinBottom, NUM_CHROMA_FORMAT, m_isTopFieldFirst );
      }

      const AccessUnit& auTop = *(iterBitstream++);
      const vector<UInt>& statsTop = writeAnnexB(bitstreamFile, auTop);
      rateStatsAccum(auTop, statsTop);

      const AccessUnit& auBottom = *(iterBitstream++);
      const vector<UInt>& statsBottom = writeAnnexB(bitstreamFile, auBottom);
      rateStatsAccum(auBottom, statsBottom);
    }
  }
  else
  {
    Int i;

    TComList<TComPicYuv*>::iterator iterPicYuvRec = m_cListPicYuvRec.end();
    list<AccessUnit>::const_iterator iterBitstream = accessUnits.begin();

    for ( i = 0; i < iNumEncoded; i++ )
    {
      --iterPicYuvRec;
    }

    for ( i = 0; i < iNumEncoded; i++ )
    {
      TComPicYuv*  pcPicYuvRec  = *(iterPicYuvRec++);
      if (m_pchReconFile)
      {
        m_cTVideoIOYuvReconFile.write( pcPicYuvRec, ipCSC, m_confWinLeft, m_confWinRight, m_confWinTop, m_confWinBottom,
            NUM_CHROMA_FORMAT, m_bClipOutputVideoToRec709Range  );
      }

      const AccessUnit& au = *(iterBitstream++);
      const vector<UInt>& stats = writeAnnexB(bitstreamFile, au);
      rateStatsAccum(au, stats);
    }
  }
}

/**
 *
 */
Void TAppEncTop::rateStatsAccum(const AccessUnit& au, const std::vector<UInt>& annexBsizes)
{
  AccessUnit::const_iterator it_au = au.begin();
  vector<UInt>::const_iterator it_stats = annexBsizes.begin();

  for (; it_au != au.end(); it_au++, it_stats++)
  {
    switch ((*it_au)->m_nalUnitType)
    {
    case NAL_UNIT_CODED_SLICE_TRAIL_R:
    case NAL_UNIT_CODED_SLICE_TRAIL_N:
    case NAL_UNIT_CODED_SLICE_TSA_R:
    case NAL_UNIT_CODED_SLICE_TSA_N:
    case NAL_UNIT_CODED_SLICE_STSA_R:
    case NAL_UNIT_CODED_SLICE_STSA_N:
    case NAL_UNIT_CODED_SLICE_BLA_W_LP:
    case NAL_UNIT_CODED_SLICE_BLA_W_RADL:
    case NAL_UNIT_CODED_SLICE_BLA_N_LP:
    case NAL_UNIT_CODED_SLICE_IDR_W_RADL:
    case NAL_UNIT_CODED_SLICE_IDR_N_LP:
    case NAL_UNIT_CODED_SLICE_CRA:
    case NAL_UNIT_CODED_SLICE_RADL_N:
    case NAL_UNIT_CODED_SLICE_RADL_R:
    case NAL_UNIT_CODED_SLICE_RASL_N:
    case NAL_UNIT_CODED_SLICE_RASL_R:
    case NAL_UNIT_VPS:
    case NAL_UNIT_SPS:
    case NAL_UNIT_PPS:
      m_essentialBytes += *it_stats;
      break;
    default:
      break;
    }

    m_totalBytes += *it_stats;
  }
}

Void TAppEncTop::printRateSummary()
{
  Double time = (Double) m_iFrameRcvd / m_iFrameRate;
  printf("Bytes written to file: %u (%.3f kbps)\n", m_totalBytes, 0.008 * m_totalBytes / time);
  if (m_summaryVerboseness > 0)
  {
    printf("Bytes for SPS/PPS/Slice (Incl. Annex B): %u (%.3f kbps)\n", m_essentialBytes, 0.008 * m_essentialBytes / time);
  }
}

Void TAppEncTop::printChromaFormat()
{
  std::cout << std::setw(43) << "Input ChromaFormatIDC = ";
  switch (m_InputChromaFormatIDC)
  {
  case CHROMA_400:  std::cout << "  4:0:0"; break;
  case CHROMA_420:  std::cout << "  4:2:0"; break;
  case CHROMA_422:  std::cout << "  4:2:2"; break;
  case CHROMA_444:  std::cout << "  4:4:4"; break;
  default:
    std::cerr << "Invalid";
    exit(1);
  }
  std::cout << std::endl;

  std::cout << std::setw(43) << "Output (internal) ChromaFormatIDC = ";
  switch (m_cTEncTop.getChromaFormatIdc())
  {
  case CHROMA_400:  std::cout << "  4:0:0"; break;
  case CHROMA_420:  std::cout << "  4:2:0"; break;
  case CHROMA_422:  std::cout << "  4:2:2"; break;
  case CHROMA_444:  std::cout << "  4:4:4"; break;
  default:
    std::cerr << "Invalid";
    exit(1);
  }
  std::cout << "\n" << std::endl;
}

//! \}
