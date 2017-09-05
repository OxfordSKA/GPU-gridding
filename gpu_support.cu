/****************************************************************************************
 *
 * Copyright the Numerical Algorithms Group Ltd, 2015.
 *
 * Created by the Numerical Algorithms Group Ltd, 2015.
 * 
 ***************************************************************************************/


#include "gpu_support.hpp"

void (*ErrorHandler::cudaErrorHandler)(cudaError_t, const char*, int)  = NULL;
void (*ErrorHandler::cublasErrorHandler)(cublasStatus_t,const char*,int)= NULL;
void (*ErrorHandler::otherErrorHandler)(const char *,const char*,int)= NULL;

