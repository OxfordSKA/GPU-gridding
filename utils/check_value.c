
#include <math.h>
#include <stdio.h>

int check_value_float(float a, float b)
{
    const float xover = 0.1;
    const float eps = 5e-1;

    const float fa = fabs(a), fb = fabs(b);
    /*const float smallest = (fa < fb ? fa : fb);*/
    const float largest  = (fa < fb ? fb : fa);

    /* If both values are small, use absolute difference */
    if (largest < xover)
    {
        const float diff = fabs(a - b);
        if (diff < eps)
            return 1;
        else
        {
            printf("check_value_float sees %.7g, \t%.7g \t: abs diff=%g\n", a, b, diff);
            return 0;
        }
    }
    else
    {
        /* At least one value is big, use relative difference */
        const float diff = fabs(a-b) / largest;
        if (diff < eps)
            return 1;
        else
        {
            printf("check_value_float sees %.7g, \t%.7g \t: rel diff=%g\n", a, b, diff);
            return 0;
        }
    }
}

int check_value_double(double a, double b)
{
    const double xover = 0.1;
    const double eps = 1e-10;

    const double fa = fabs(a), fb = fabs(b);
    /*const double smallest = (fa < fb ? fa : fb);*/
    const double largest  = (fa < fb ? fb : fa);

    /* If both values are small, use absolute difference */
    if (largest < xover)
    {
        const double diff = fabs(a - b);
        if (diff < eps)
            return 1;
        else
        {
            printf("check_value_double sees %.7g, \t%.7g \t: abs diff=%g\n", a, b, diff);
            return 0;
        }
    }
    else
    {
        /* At least one value is big, use relative difference */
        const double diff = fabs(a-b) / largest;
        if (diff < eps)
            return 1;
        else
        {
            printf("check_value_double sees %.7g, \t%.7g \t: rel diff=%g\n", a, b, diff);
            return 0;
        }
    }
}
