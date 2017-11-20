#pragma once

#ifdef __CUDACC__
#define DECL __host__ __device__
#else
#define DECL 
#endif


struct Point {
   int u;
   int v;

   DECL Point(int u, int v) : u(u), v(v) {}
   DECL Point(const Point& copy) : u(copy.u), v(copy.v) { }

   friend Point operator+(Point lhs, const Point &rhs) {
      lhs.u += rhs.u;
      lhs.v += rhs.v;
      return lhs;
   }

   friend Point operator-(Point lhs, const Point &rhs) {
      lhs.u -= rhs.u;
      lhs.v -= rhs.v;
      return lhs;
   }

   Point & operator+=(const Point &rhs) {
      u += rhs.u;
      v += rhs.v;
      return *this;
   }

   Point & operator-=(const Point &rhs) {
      u -= rhs.u;
      v -= rhs.v;
      return *this;
   }

   Point & operator*=(float x) {
      u = int( x * u );
      v = int( x * v );
      return *this;
   }

   friend Point operator*(Point lhs, float x) {
      lhs *= x;
      return lhs;
   }
   friend Point operator*(float x, Point rhs) {
      rhs *= x;
      return rhs;
   }

};


struct Box {
   Point topLeft;
   Point botRight;


   // Creates an empty region
   DECL Box() : topLeft(0,0), botRight(-1, -1) { }

   DECL Box(const Point &tl, const Point &br) : topLeft(tl), botRight(br) { }

   DECL bool isEmpty() {
      return (topLeft.u > botRight.u) || (topLeft.v > botRight.v);
   }

};





