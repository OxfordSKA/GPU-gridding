#pragma once

#include <assert.h>

extern "C" {
#include "bmpfile.h"
}


#include "utils.hpp"


/**
 * Simple routine to write a BMP file in black and white using the 
 * BMPFILE package which is freely available on the internet.
 *
 * Inputs:
 *    filename    -   name of file to create
 *      B         -   array of 0s and 1s representing B&W image
 *    imgWidt     -   width of image (B)
 *  imgHeight     -   height of image (B)
 */
template<typename FP>
void writeBmp(const char * filename, const FP * B, const int imgWidth, const int imgHeight)
{
   bmpfile_t * bmp = NULL;
   rgb_pixel_t white = {255,255,255,0};
   rgb_pixel_t black = {0,0,0,0};
   bmp = bmp_create(imgWidth,imgHeight,1);
   assert(bmp);

   for(int y=0; y<imgHeight; y++) {
      for(int x=0; x<imgWidth; x++) {
         if(B[y*imgWidth+x])
            bmp_set_pixel(bmp,x,y,white);
         else
            bmp_set_pixel(bmp,x,y,black);
      }
   }
   assert(bmp_save(bmp,filename));
   bmp_destroy(bmp);
}

/**
 * Simple routine to write a BMP file in grayscale using the 
 * BMPFILE package which is freely available on the internet.
 *
 * Inputs:
 *    filename    -   name of file to create
 *      B         -   array of unsigned integers representing intensities
 *    imgWidt     -   width of image (B)
 *  imgHeight     -   height of image (B)
 */
template<typename FP>
void writeBmpGrayscale(const char * filename, const FP * B, const int imgWidth, const int imgHeight)
{
   bmpfile_t * bmp = NULL;
   
   const uint8_t maxgray = 255;
   const uint8_t mingray = 25;
   bmp = bmp_create(imgWidth,imgHeight,24);
   assert(bmp);

   // Get the maximal value so that we can scale
   FP maxval = 0;
   for(unsigned int i=0; i<imgWidth*imgHeight; i++) {
      FP val = B[i];
      maxval = (maxval < val ? val : maxval);
   }

   for(int y=0; y<imgHeight; y++) {
      for(int x=0; x<imgWidth; x++) {
         FP val = B[y*imgWidth+x];
         uint8_t gray = 0;
         if( val > 0) {
            double frac = double(val-1)/double(maxval);
            gray = (uint8_t)( mingray + frac * (maxgray-mingray) );
         }
         rgb_pixel_t pix = {gray, gray, gray, 0};
         bmp_set_pixel(bmp,x,y,pix);
      }
   }
   assert(bmp_save(bmp,filename));
   bmp_destroy(bmp);
}



template<typename FP>
void writeBmpGrayscaleWithBoxes(const char * filename, const FP * B, 
      const int topx,
      const int topy,
      const int botx,
      const int boty,
      const int imgWidth, const int imgHeight)
{
   bmpfile_t * bmp = NULL;
   
   int m = (botx - topx)/3;
   const int xbox[] = {topx, botx, topx+m, topx+2*m};
   m = (boty - topy)/3;
   const int ybox[] = {topy, boty, topy+m, topy+2*m};
   const int nbox = 4;

   const uint8_t maxgray = 255;
   const uint8_t mingray = 25;
   bmp = bmp_create(imgWidth,imgHeight,24);
   assert(bmp);

   // Get the maximal value so that we can scale
   FP maxval = 0;
   for(unsigned int i=0; i<imgWidth*imgHeight; i++) {
      FP val = B[i];
      maxval = (maxval < val ? val : maxval);
   }

   for(int y=0; y<imgHeight; y++) {
      for(int x=0; x<imgWidth; x++) {
         FP val = B[y*imgWidth+x];
         uint8_t gray = 0;
         bool box = false;
         for(int i=0; i<nbox; i++) {
            box = box || (x==xbox[i]);
            box = box || (y==ybox[i]);
         }
         if(box) {
            // Set to white
            gray = 255;
         }
         else if( val > 0) {
            double frac = double(val-1)/double(maxval);
            gray = (uint8_t)( mingray + frac * (maxgray-mingray) );
         }
         rgb_pixel_t pix = {gray, gray, gray, 0};
         bmp_set_pixel(bmp,x,y,pix);
      }
   }
   assert(bmp_save(bmp,filename));
   bmp_destroy(bmp);
}




namespace {


   float gamma_correct(float u) 
   {
      // Standard CRT Gamma
      const float GAMMA = 2.4f;
      if( u > 0.00304f )
	return 1.055f*pow(u, 1.0f/GAMMA) - 0.055f;
      else
	return 12.92f*u;
   }


   rgb_pixel_t hcl2rgb(float h, float c, float l)
   {
      // D65 White Point
      const float WHITE_Y = 100.000f;
      const float WHITE_u = 0.1978398f;
      const float WHITE_v = 0.4683363f;
      const float PI = 3.1415926536f;

      if( l < 0 || l > WHITE_Y || c < 0)
         assert(false);

      // First convert to CIELUV (just a polar to Cartesian coordinate transformation)
      float L = l;
      float U = c * cos(h * PI/180.0f);
      float V = c * sin(h * PI/180.0f);

      float X, Y, Z;
      // Now convert to CIEXYZ
      if( L <= 0 && U == 0 && V == 0) {
         X = 0;
         Y = 0;
         Z = 0;
      } else {
         Y = WHITE_Y;
         if( L > 7.999592f ) {
            Y = Y*pow( (L + 16)/116.0f, 3);
         } else {
            Y = Y*L/903.3f;
         }
         float u = U/(13*L) + WHITE_u;
         float v = V/(13*L) + WHITE_v;
         X = (9.0*Y*u)/(4*v);
         Z = -X/3 - 5*Y + 3*Y/v;
      }

      // Now convert to sRGB
      float r = gamma_correct((3.240479f*X - 1.537150f*Y - 0.498535f*Z)/WHITE_Y);
      float g = gamma_correct((-0.969256f*X + 1.875992f*Y + 0.041556f*Z)/WHITE_Y);
      float b = gamma_correct((0.055648f*X - 0.204043f*Y + 1.057311f*Z)/WHITE_Y);

      // Round to integers and correct
      r = round(255 * r);
      g = round(255 * g);
      b = round(255 * b);
      r = (r > 255 ? 255 : r);
      r = (r <  0  ?  0  : r);
      g = (g > 255 ? 255 : g);
      g = (g < 0   ? 0   : g);
      b = (b > 255 ? 255 : b);
      b = (b <  0  ?  0  : b);

      rgb_pixel_t pix = {uint8_t(r), uint8_t(g), uint8_t(b), 255};
      return pix;
   }
}


template<typename FP>
void writeBmpColor(const char * filename, const FP * B, 
      const Box box,
      const int imgWidth, const int imgHeight)
{
   bmpfile_t * bmp = NULL;
   
   std::vector<int> panelCoords_x = {box.topLeft.u, box.botRight.u};
   std::vector<int> panelCoords_y = {box.topLeft.v, box.botRight.v};
   rgb_pixel_t boxColor = {255, 255, 255, 0};

   bmp = bmp_create(imgWidth,imgHeight,24);
   assert(bmp);

   // Get the maximal value so that we can scale
   FP maxval = 0;
   for(unsigned int i=0; i<imgWidth*imgHeight; i++) {
      FP val = B[i];
      maxval = (maxval < val ? val : maxval);
   }
   printf("BMP WRITE: found maxval=%d.  Be sure to include this in report so scale is clear\n", maxval);

   const float HCLChroma = 80.0f;
   const float HCLHue = 0.0f;
   const int colourmapsz = 1000;
   std::vector<rgb_pixel_t> colourmap(colourmapsz);
   for(int i=0; i<colourmapsz; i++) {
      float lum = ( float(i+1)/float(colourmapsz) )*100.0f;
      colourmap.at(i) = hcl2rgb(HCLHue, HCLChroma, lum);
   }


   for(int y=0; y<imgHeight; y++) {
      for(int x=0; x<imgWidth; x++) {
         FP val = B[y*imgWidth+x];

         bool box = false;
         for(int i=0; i<panelCoords_x.size(); i++) {
            box = box || (x==panelCoords_x[i]);
            box = box || (y==panelCoords_y[i]);
         }
         rgb_pixel_t pix = {0,0,0,255};

         if(box) {
            pix = boxColor;            
         }
         else if(val>0) {
            int idx = int( ( float(val)/float(maxval+1) ) * colourmapsz );
            pix = colourmap.at(idx);
         }
         bmp_set_pixel(bmp,x,y,pix);
      }
   }

   // Draw colour map bar
   {
      int rskip = int( 0.5f * (box.botRight.u-box.topLeft.u) );
      int height = (box.botRight.v - box.topLeft.v + colourmapsz-1) / colourmapsz;
      int width = int( 0.1f*(box.botRight.u-box.topLeft.u) );
      
      for(int i=0; i<colourmapsz; i++) {
         int startx = box.botRight.u + rskip;
         int endx   = startx + width;
         int starty = box.botRight.v - (i+1)*height;
         int endy   = starty + height;

         for(int x=startx; x<=endx; x++) {
            for(int y=starty; y<=endy; y++) {
               bmp_set_pixel(bmp, x, y, colourmap.at(i));
            }
         }
      }
   }
   assert(bmp_save(bmp,filename));
   bmp_destroy(bmp);
}



template<typename FP>
void writeBmpColorWithTiles(const char * filename, const FP * B, 
      const Box box, const Point numTiles, const Point tileSize,
      const int gridWidth, const int gridHeight, bool drawTiles=true)
{


   bmpfile_t * bmp = NULL;
   
   std::vector<int> panelCoords_x, panelCoords_y;
   for(int i=0; i<=numTiles.u; i++) {
      panelCoords_x.push_back(box.topLeft.u + i*tileSize.u);
   }
   for(int i=0; i<=numTiles.v; i++) {
      panelCoords_y.push_back(box.topLeft.v + i*tileSize.v);
   }
   rgb_pixel_t boxColor = {255, 255, 255, 0};


   // Get the maximal value so that we can scale
   FP maxval = 0;
   for(unsigned int i=0; i<gridWidth*gridHeight; i++) {
      FP val = B[i];
      maxval = (maxval < val ? val : maxval);
   }
   printf("BMP WRITE: found maxval=%d.  Grid is %dx%d and box is (%d,%d) x (%d,%d)."
         "Be sure to include this in report so scale is clear\n", maxval, gridWidth, gridHeight, 
         box.topLeft.u, box.topLeft.v, box.botRight.u, box.botRight.v);

   const float HCLChroma = 80.0f;
   const float HCLHue = 0.0f;
   const int colourmapsz = 1000;
   std::vector<rgb_pixel_t> colourmap(colourmapsz);
   for(int i=0; i<colourmapsz; i++) {
      float lum = ( float(i+1)/float(colourmapsz) )*100.0f;
      colourmap.at(i) = hcl2rgb(HCLHue, HCLChroma, lum);
   }

   const int rskip = int( 0.1f * (box.botRight.u-box.topLeft.u) );
   const int height = (box.botRight.v - box.topLeft.v + colourmapsz-1) / colourmapsz;
   const int width = int( 0.1f*(box.botRight.u-box.topLeft.u) );

   bmp = bmp_create(box.botRight.u-box.topLeft.u+1+rskip+width+50, 
         std::max(box.botRight.v-box.topLeft.v+50,height*colourmapsz+50), 24);
   assert(bmp);


   for(int y=0; y<gridHeight; y++) {
      for(int x=0; x<gridWidth; x++) {
         FP val = B[y*gridWidth+x];

         bool panelBoundary = false;
         if(drawTiles) {
            for(int i=0; i<panelCoords_x.size(); i++) {
               panelBoundary = panelBoundary || (x==panelCoords_x[i]);
            }
            for(int i=0; i<panelCoords_y.size(); i++) {
               panelBoundary = panelBoundary || (y==panelCoords_y[i]);
            }
         }
         rgb_pixel_t pix = {0,0,0,255};

         if(panelBoundary) {
            pix = boxColor;            
         }
         else if(val>0) {
            int idx = int( ( float(val)/float(maxval+1) ) * colourmapsz );
            pix = colourmap.at(idx);
         }
         if(x>=box.topLeft.u && y>=box.topLeft.v &&
               x<=box.botRight.u && y<= box.botRight.v) {
            bmp_set_pixel(bmp, x-box.topLeft.u, y-box.topLeft.v, pix);
         }
      }
   }


   // Draw colour map bar
   {
      for(int i=0; i<colourmapsz; i++) {
         int startx = box.botRight.u + rskip;
         int endx   = startx + width;
         int starty = box.botRight.v - (i+1)*height;
         int endy   = starty + height;

         for(int x=startx; x<=endx; x++) {
            for(int y=starty; y<=endy; y++) {
               bmp_set_pixel(bmp, x-box.topLeft.u, y-box.topLeft.v, colourmap.at(i));
            }
         }
      }
   }
   assert(bmp_save(bmp,filename));
   bmp_destroy(bmp);
}


