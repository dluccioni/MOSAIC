
#include <Python.h>
#include <stddef.h>

/* this block of #ifs should be kept exactly identical between
   c/_cffi_backend.c, cffi/vengine_cpy.py, cffi/vengine_gen.py
   and cffi/_cffi_include.h */
#if defined(_MSC_VER)
# include <malloc.h>   /* for alloca() */
# if _MSC_VER < 1600   /* MSVC < 2010 */
   typedef __int8 int8_t;
   typedef __int16 int16_t;
   typedef __int32 int32_t;
   typedef __int64 int64_t;
   typedef unsigned __int8 uint8_t;
   typedef unsigned __int16 uint16_t;
   typedef unsigned __int32 uint32_t;
   typedef unsigned __int64 uint64_t;
   typedef __int8 int_least8_t;
   typedef __int16 int_least16_t;
   typedef __int32 int_least32_t;
   typedef __int64 int_least64_t;
   typedef unsigned __int8 uint_least8_t;
   typedef unsigned __int16 uint_least16_t;
   typedef unsigned __int32 uint_least32_t;
   typedef unsigned __int64 uint_least64_t;
   typedef __int8 int_fast8_t;
   typedef __int16 int_fast16_t;
   typedef __int32 int_fast32_t;
   typedef __int64 int_fast64_t;
   typedef unsigned __int8 uint_fast8_t;
   typedef unsigned __int16 uint_fast16_t;
   typedef unsigned __int32 uint_fast32_t;
   typedef unsigned __int64 uint_fast64_t;
   typedef __int64 intmax_t;
   typedef unsigned __int64 uintmax_t;
# else
#  include <stdint.h>
# endif
# if _MSC_VER < 1800   /* MSVC < 2013 */
#  ifndef __cplusplus
    typedef unsigned char _Bool;
#  endif
# endif
# define _cffi_float_complex_t   _Fcomplex    /* include <complex.h> for it */
# define _cffi_double_complex_t  _Dcomplex    /* include <complex.h> for it */
#else
# include <stdint.h>
# if (defined (__SVR4) && defined (__sun)) || defined(_AIX) || defined(__hpux)
#  include <alloca.h>
# endif
# define _cffi_float_complex_t   float _Complex
# define _cffi_double_complex_t  double _Complex
#endif

#if PY_MAJOR_VERSION < 3
# undef PyCapsule_CheckExact
# undef PyCapsule_GetPointer
# define PyCapsule_CheckExact(capsule) (PyCObject_Check(capsule))
# define PyCapsule_GetPointer(capsule, name) \
    (PyCObject_AsVoidPtr(capsule))
#endif

#if PY_MAJOR_VERSION >= 3
# define PyInt_FromLong PyLong_FromLong
#endif

#define _cffi_from_c_double PyFloat_FromDouble
#define _cffi_from_c_float PyFloat_FromDouble
#define _cffi_from_c_long PyInt_FromLong
#define _cffi_from_c_ulong PyLong_FromUnsignedLong
#define _cffi_from_c_longlong PyLong_FromLongLong
#define _cffi_from_c_ulonglong PyLong_FromUnsignedLongLong
#define _cffi_from_c__Bool PyBool_FromLong

#define _cffi_to_c_double PyFloat_AsDouble
#define _cffi_to_c_float PyFloat_AsDouble

#define _cffi_from_c_int_const(x)                                        \
    (((x) > 0) ?                                                         \
        ((unsigned long long)(x) <= (unsigned long long)LONG_MAX) ?      \
            PyInt_FromLong((long)(x)) :                                  \
            PyLong_FromUnsignedLongLong((unsigned long long)(x)) :       \
        ((long long)(x) >= (long long)LONG_MIN) ?                        \
            PyInt_FromLong((long)(x)) :                                  \
            PyLong_FromLongLong((long long)(x)))

#define _cffi_from_c_int(x, type)                                        \
    (((type)-1) > 0 ? /* unsigned */                                     \
        (sizeof(type) < sizeof(long) ?                                   \
            PyInt_FromLong((long)x) :                                    \
         sizeof(type) == sizeof(long) ?                                  \
            PyLong_FromUnsignedLong((unsigned long)x) :                  \
            PyLong_FromUnsignedLongLong((unsigned long long)x)) :        \
        (sizeof(type) <= sizeof(long) ?                                  \
            PyInt_FromLong((long)x) :                                    \
            PyLong_FromLongLong((long long)x)))

#define _cffi_to_c_int(o, type)                                          \
    ((type)(                                                             \
     sizeof(type) == 1 ? (((type)-1) > 0 ? (type)_cffi_to_c_u8(o)        \
                                         : (type)_cffi_to_c_i8(o)) :     \
     sizeof(type) == 2 ? (((type)-1) > 0 ? (type)_cffi_to_c_u16(o)       \
                                         : (type)_cffi_to_c_i16(o)) :    \
     sizeof(type) == 4 ? (((type)-1) > 0 ? (type)_cffi_to_c_u32(o)       \
                                         : (type)_cffi_to_c_i32(o)) :    \
     sizeof(type) == 8 ? (((type)-1) > 0 ? (type)_cffi_to_c_u64(o)       \
                                         : (type)_cffi_to_c_i64(o)) :    \
     (Py_FatalError("unsupported size for type " #type), (type)0)))

#define _cffi_to_c_i8                                                    \
                 ((int(*)(PyObject *))_cffi_exports[1])
#define _cffi_to_c_u8                                                    \
                 ((int(*)(PyObject *))_cffi_exports[2])
#define _cffi_to_c_i16                                                   \
                 ((int(*)(PyObject *))_cffi_exports[3])
#define _cffi_to_c_u16                                                   \
                 ((int(*)(PyObject *))_cffi_exports[4])
#define _cffi_to_c_i32                                                   \
                 ((int(*)(PyObject *))_cffi_exports[5])
#define _cffi_to_c_u32                                                   \
                 ((unsigned int(*)(PyObject *))_cffi_exports[6])
#define _cffi_to_c_i64                                                   \
                 ((long long(*)(PyObject *))_cffi_exports[7])
#define _cffi_to_c_u64                                                   \
                 ((unsigned long long(*)(PyObject *))_cffi_exports[8])
#define _cffi_to_c_char                                                  \
                 ((int(*)(PyObject *))_cffi_exports[9])
#define _cffi_from_c_pointer                                             \
    ((PyObject *(*)(char *, CTypeDescrObject *))_cffi_exports[10])
#define _cffi_to_c_pointer                                               \
    ((char *(*)(PyObject *, CTypeDescrObject *))_cffi_exports[11])
#define _cffi_get_struct_layout                                          \
    ((PyObject *(*)(Py_ssize_t[]))_cffi_exports[12])
#define _cffi_restore_errno                                              \
    ((void(*)(void))_cffi_exports[13])
#define _cffi_save_errno                                                 \
    ((void(*)(void))_cffi_exports[14])
#define _cffi_from_c_char                                                \
    ((PyObject *(*)(char))_cffi_exports[15])
#define _cffi_from_c_deref                                               \
    ((PyObject *(*)(char *, CTypeDescrObject *))_cffi_exports[16])
#define _cffi_to_c                                                       \
    ((int(*)(char *, CTypeDescrObject *, PyObject *))_cffi_exports[17])
#define _cffi_from_c_struct                                              \
    ((PyObject *(*)(char *, CTypeDescrObject *))_cffi_exports[18])
#define _cffi_to_c_wchar_t                                               \
    ((wchar_t(*)(PyObject *))_cffi_exports[19])
#define _cffi_from_c_wchar_t                                             \
    ((PyObject *(*)(wchar_t))_cffi_exports[20])
#define _cffi_to_c_long_double                                           \
    ((long double(*)(PyObject *))_cffi_exports[21])
#define _cffi_to_c__Bool                                                 \
    ((_Bool(*)(PyObject *))_cffi_exports[22])
#define _cffi_prepare_pointer_call_argument                              \
    ((Py_ssize_t(*)(CTypeDescrObject *, PyObject *, char **))_cffi_exports[23])
#define _cffi_convert_array_from_object                                  \
    ((int(*)(char *, CTypeDescrObject *, PyObject *))_cffi_exports[24])
#define _CFFI_NUM_EXPORTS 25

typedef struct _ctypedescr CTypeDescrObject;

static void *_cffi_exports[_CFFI_NUM_EXPORTS];
static PyObject *_cffi_types, *_cffi_VerificationError;

static int _cffi_setup_custom(PyObject *lib);   /* forward */

static PyObject *_cffi_setup(PyObject *self, PyObject *args)
{
    PyObject *library;
    int was_alive = (_cffi_types != NULL);
    (void)self; /* unused */
    if (!PyArg_ParseTuple(args, "OOO", &_cffi_types, &_cffi_VerificationError,
                                       &library))
        return NULL;
    Py_INCREF(_cffi_types);
    Py_INCREF(_cffi_VerificationError);
    if (_cffi_setup_custom(library) < 0)
        return NULL;
    return PyBool_FromLong(was_alive);
}

union _cffi_union_alignment_u {
    unsigned char m_char;
    unsigned short m_short;
    unsigned int m_int;
    unsigned long m_long;
    unsigned long long m_longlong;
    float m_float;
    double m_double;
    long double m_longdouble;
};

struct _cffi_freeme_s {
    struct _cffi_freeme_s *next;
    union _cffi_union_alignment_u alignment;
};

#ifdef __GNUC__
  __attribute__((unused))
#endif
static int _cffi_convert_array_argument(CTypeDescrObject *ctptr, PyObject *arg,
                                        char **output_data, Py_ssize_t datasize,
                                        struct _cffi_freeme_s **freeme)
{
    char *p;
    if (datasize < 0)
        return -1;

    p = *output_data;
    if (p == NULL) {
        struct _cffi_freeme_s *fp = (struct _cffi_freeme_s *)PyObject_Malloc(
            offsetof(struct _cffi_freeme_s, alignment) + (size_t)datasize);
        if (fp == NULL)
            return -1;
        fp->next = *freeme;
        *freeme = fp;
        p = *output_data = (char *)&fp->alignment;
    }
    memset((void *)p, 0, (size_t)datasize);
    return _cffi_convert_array_from_object(p, ctptr, arg);
}

#ifdef __GNUC__
  __attribute__((unused))
#endif
static void _cffi_free_array_arguments(struct _cffi_freeme_s *freeme)
{
    do {
        void *p = (void *)freeme;
        freeme = freeme->next;
        PyObject_Free(p);
    } while (freeme != NULL);
}

static int _cffi_init(void)
{
    PyObject *module, *c_api_object = NULL;

    module = PyImport_ImportModule("_cffi_backend");
    if (module == NULL)
        goto failure;

    c_api_object = PyObject_GetAttrString(module, "_C_API");
    if (c_api_object == NULL)
        goto failure;
    if (!PyCapsule_CheckExact(c_api_object)) {
        PyErr_SetNone(PyExc_ImportError);
        goto failure;
    }
    memcpy(_cffi_exports, PyCapsule_GetPointer(c_api_object, "cffi"),
           _CFFI_NUM_EXPORTS * sizeof(void *));

    Py_DECREF(module);
    Py_DECREF(c_api_object);
    return 0;

  failure:
    Py_XDECREF(module);
    Py_XDECREF(c_api_object);
    return -1;
}

#define _cffi_type(num) ((CTypeDescrObject *)PyList_GET_ITEM(_cffi_types, num))

/**********/



        #include <math.h>
        #include <stdlib.h> // for malloc/free if needed
        // Dot product
        static double dot3(const double *a, const double *b){
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
        }

        // Cross product out = a x b
        static void cross3(const double *a, const double *b, double *out){
            out[0] = a[1]*b[2] - a[2]*b[1];
            out[1] = a[2]*b[0] - a[0]*b[2];
            out[2] = a[0]*b[1] - a[1]*b[0];
        }

        // Norm of a 3D vector
        static double norm3(const double *v){
            return sqrt(dot3(v,v));
        }

        // Project 8 points onto an axis
        // out[0] = min, out[1] = max of the projection
        static void project_points(const double *pts8x3, const double *axis, double eps, double *out){
            double axis_len = norm3(axis);
            if(axis_len < eps){
                // Degenerate axis -> all points project to zero
                out[0] = 0.0; 
                out[1] = 0.0;
                return;
            }
            double ax[3] = { axis[0]/axis_len, axis[1]/axis_len, axis[2]/axis_len };

            double val = dot3(pts8x3, ax); // first corner
            double minv = val, maxv = val;
            for(int i=1; i<8; i++){
                val = dot3(pts8x3 + 3*i, ax);
                if(val < minv) minv = val;
                if(val > maxv) maxv = val;
            }
            out[0] = minv;
            out[1] = maxv;
        }

        // Check if intervals [a0,a1] and [b0,b1] overlap
        static int intervals_overlap(const double *a, const double *b){
            // If one interval is strictly to the left of the other, no overlap
            if(a[1] < b[0] || b[1] < a[0]) 
                return 0;
            return 1;
        }

        // single_intersect: checks intersection for one pair of parallelepipeds
        // pts1, pts2 each has 8 corners -> 24 doubles
        static int single_intersect(const double *pts1, const double *pts2, double eps)
        {
            // 1) Identify shape1 edges from the known corner ordering
            //    c1 = pts1[0], e1 = pts1[1] - pts1[0], e2 = pts1[2] - pts1[0], e3 = pts1[3] - pts1[0].
            double c1[3]  = { pts1[0], pts1[1], pts1[2] };
            double e1[3]  = { pts1[3] - c1[0], pts1[4] - c1[1], pts1[5] - c1[2] };
            double e2[3]  = { pts1[6] - c1[0], pts1[7] - c1[1], pts1[8] - c1[2] };
            double e3[3]  = { pts1[9] - c1[0], pts1[10] - c1[1], pts1[11] - c1[2] };

            // 2) Identify shape2 edges similarly
            double c2[3]  = { pts2[0], pts2[1], pts2[2] };
            double f1[3]  = { pts2[3] - c2[0], pts2[4] - c2[1], pts2[5] - c2[2] };
            double f2[3]  = { pts2[6] - c2[0], pts2[7] - c2[1], pts2[8] - c2[2] };
            double f3[3]  = { pts2[9] - c2[0], pts2[10] - c2[1], pts2[11] - c2[2] };

            // 3) Rebuild all 8 corners for shape1 (so we can do standard projection)
            //    shape1[i] = c1 + alpha1 * e1 + alpha2 * e2 + alpha3 * e3,
            //    where alphaN is either 0 or 1. The corner ordering matches get_unit_corners().
            double shape1[24];
            for(int i=0; i<8; i++){
                int a1 = (i & 1) ? 1 : 0; // bit 0
                int a2 = (i & 2) ? 1 : 0; // bit 1
                int a3 = (i & 4) ? 1 : 0; // bit 2
                shape1[3*i + 0] = c1[0] + a1*e1[0] + a2*e2[0] + a3*e3[0];
                shape1[3*i + 1] = c1[1] + a1*e1[1] + a2*e2[1] + a3*e3[1];
                shape1[3*i + 2] = c1[2] + a1*e1[2] + a2*e2[2] + a3*e3[2];
            }

            // 4) Rebuild all 8 corners for shape2
            double shape2[24];
            for(int i=0; i<8; i++){
                int a1 = (i & 1) ? 1 : 0;
                int a2 = (i & 2) ? 1 : 0;
                int a3 = (i & 4) ? 1 : 0;
                shape2[3*i + 0] = c2[0] + a1*f1[0] + a2*f2[0] + a3*f3[0];
                shape2[3*i + 1] = c2[1] + a1*f1[1] + a2*f2[1] + a3*f3[1];
                shape2[3*i + 2] = c2[2] + a1*f1[2] + a2*f2[2] + a3*f3[2];
            }

            // 5) Compute the 15 candidate axes:
            //    -- 3 face normals from shape1
            //    -- 3 face normals from shape2
            //    -- 9 cross products of edges from shape1 x edges from shape2

            // shape1 face normals
            double n1[3], n2[3], n3[3];
            cross3(e1, e2, n1);
            cross3(e2, e3, n2);
            cross3(e3, e1, n3);

            // shape2 face normals
            double m1[3], m2[3], m3[3];
            cross3(f1, f2, m1);
            cross3(f2, f3, m2);
            cross3(f3, f1, m3);

            double edges1[3][3] = {{e1[0], e1[1], e1[2]},
                                {e2[0], e2[1], e2[2]},
                                {e3[0], e3[1], e3[2]}};
            double edges2[3][3] = {{f1[0], f1[1], f1[2]},
                                {f2[0], f2[1], f2[2]},
                                {f3[0], f3[1], f3[2]}};

            // Fill up to 15 axes in an array
            double axes[15][3];
            int axisCount = 0;

            // shape1 face normals
            axes[axisCount][0] = n1[0]; axes[axisCount][1] = n1[1]; axes[axisCount][2] = n1[2]; axisCount++;
            axes[axisCount][0] = n2[0]; axes[axisCount][1] = n2[1]; axes[axisCount][2] = n2[2]; axisCount++;
            axes[axisCount][0] = n3[0]; axes[axisCount][1] = n3[1]; axes[axisCount][2] = n3[2]; axisCount++;

            // shape2 face normals
            axes[axisCount][0] = m1[0]; axes[axisCount][1] = m1[1]; axes[axisCount][2] = m1[2]; axisCount++;
            axes[axisCount][0] = m2[0]; axes[axisCount][1] = m2[1]; axes[axisCount][2] = m2[2]; axisCount++;
            axes[axisCount][0] = m3[0]; axes[axisCount][1] = m3[1]; axes[axisCount][2] = m3[2]; axisCount++;

            // cross products of edges
            for(int i=0; i<3; i++){
                for(int j=0; j<3; j++){
                    double c12[3];
                    cross3(edges1[i], edges2[j], c12);
                    double len_c12 = norm3(c12);
                    if(len_c12 > eps){  // skip near-degenerate
                        axes[axisCount][0] = c12[0];
                        axes[axisCount][1] = c12[1];
                        axes[axisCount][2] = c12[2];
                        axisCount++;
                    }
                }
            }

            // 6) Run the SAT test
            double proj1[2], proj2[2];
            for(int a=0; a<axisCount; a++){
                project_points(shape1, axes[a], eps, proj1);
                project_points(shape2, axes[a], eps, proj2);
                if(!intervals_overlap(proj1, proj2)){
                    // Found a separating axis -> no intersection
                    return 0;
                }
            }
            // No separating axis found => shapes intersect
            return 1;
        }

        // --------------------------------------------------------------------
        // BATCH function: parallelepipeds_intersect for n parallelepipeds
        // all_pts1: length 24*n (each block of 8 corners = 24 floats)
        // pts2    : just one shape of 8 corners = 24 floats
        // out_intersect[i] = 0 or 1
        // --------------------------------------------------------------------
        int check_parallelepipeds_intersect_batch(const double *all_pts1,
                                                const double *pts2,
                                                double eps,
                                                int n,
                                                int *out_intersect)
        {
            for(int i=0; i<n; i++){
                const double *shape_i = all_pts1 + 24*i; 
                out_intersect[i] = single_intersect(shape_i, pts2, eps);
            }
            return 0; // success
        }
        

static PyObject *
_cffi_f_check_parallelepipeds_intersect_batch(PyObject *self, PyObject *args)
{
  double const * x0;
  double const * x1;
  double x2;
  int x3;
  int * x4;
  Py_ssize_t datasize;
  struct _cffi_freeme_s *large_args_free = NULL;
  int result;
  PyObject *pyresult;
  PyObject *arg0;
  PyObject *arg1;
  PyObject *arg2;
  PyObject *arg3;
  PyObject *arg4;

  if (!PyArg_ParseTuple(args, "OOOOO:check_parallelepipeds_intersect_batch", &arg0, &arg1, &arg2, &arg3, &arg4))
    return NULL;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(0), arg0, (char **)&x0);
  if (datasize != 0) {
    x0 = ((size_t)datasize) <= 640 ? alloca((size_t)datasize) : NULL;
    if (_cffi_convert_array_argument(_cffi_type(0), arg0, (char **)&x0,
            datasize, &large_args_free) < 0)
      return NULL;
  }

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(0), arg1, (char **)&x1);
  if (datasize != 0) {
    x1 = ((size_t)datasize) <= 640 ? alloca((size_t)datasize) : NULL;
    if (_cffi_convert_array_argument(_cffi_type(0), arg1, (char **)&x1,
            datasize, &large_args_free) < 0)
      return NULL;
  }

  x2 = (double)_cffi_to_c_double(arg2);
  if (x2 == (double)-1 && PyErr_Occurred())
    return NULL;

  x3 = _cffi_to_c_int(arg3, int);
  if (x3 == (int)-1 && PyErr_Occurred())
    return NULL;

  datasize = _cffi_prepare_pointer_call_argument(
      _cffi_type(1), arg4, (char **)&x4);
  if (datasize != 0) {
    x4 = ((size_t)datasize) <= 640 ? alloca((size_t)datasize) : NULL;
    if (_cffi_convert_array_argument(_cffi_type(1), arg4, (char **)&x4,
            datasize, &large_args_free) < 0)
      return NULL;
  }

  Py_BEGIN_ALLOW_THREADS
  _cffi_restore_errno();
  { result = check_parallelepipeds_intersect_batch(x0, x1, x2, x3, x4); }
  _cffi_save_errno();
  Py_END_ALLOW_THREADS

  (void)self; /* unused */
  pyresult = _cffi_from_c_int(result, int);
  if (large_args_free != NULL) _cffi_free_array_arguments(large_args_free);
  return pyresult;
}

static int _cffi_setup_custom(PyObject *lib)
{
  return ((void)lib,0);
}

static PyMethodDef _cffi_methods[] = {
  {"check_parallelepipeds_intersect_batch", _cffi_f_check_parallelepipeds_intersect_batch, METH_VARARGS, NULL},
  {"_cffi_setup", _cffi_setup, METH_VARARGS, NULL},
  {NULL, NULL, 0, NULL}    /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef _cffi_module_def = {
  PyModuleDef_HEAD_INIT,
  "_cffi__x8ee81dbex64ee6e41",
  NULL,
  -1,
  _cffi_methods,
  NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit__cffi__x8ee81dbex64ee6e41(void)
{
  PyObject *lib;
  lib = PyModule_Create(&_cffi_module_def);
  if (lib == NULL)
    return NULL;
  if (((void)lib,0) < 0 || _cffi_init() < 0) {
    Py_DECREF(lib);
    return NULL;
  }
  return lib;
}

#else

PyMODINIT_FUNC
init_cffi__x8ee81dbex64ee6e41(void)
{
  PyObject *lib;
  lib = Py_InitModule("_cffi__x8ee81dbex64ee6e41", _cffi_methods);
  if (lib == NULL)
    return;
  if (((void)lib,0) < 0 || _cffi_init() < 0)
    return;
  return;
}

#endif
