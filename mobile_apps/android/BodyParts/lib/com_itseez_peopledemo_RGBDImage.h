/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_itseez_peopledemo_RGBDImage */

#ifndef _Included_com_itseez_peopledemo_RGBDImage
#define _Included_com_itseez_peopledemo_RGBDImage
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_itseez_peopledemo_RGBDImage
 * Method:    cacheIds
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_itseez_peopledemo_RGBDImage_cacheIds
  (JNIEnv *, jclass);

/*
 * Class:     com_itseez_peopledemo_RGBDImage
 * Method:    create
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_itseez_peopledemo_RGBDImage_create
  (JNIEnv *, jobject);

/*
 * Class:     com_itseez_peopledemo_RGBDImage
 * Method:    free
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_com_itseez_peopledemo_RGBDImage_free
  (JNIEnv *, jobject);

/*
 * Class:     com_itseez_peopledemo_RGBDImage
 * Method:    readColors
 * Signature: ([I)V
 */
JNIEXPORT void JNICALL Java_com_itseez_peopledemo_RGBDImage_readColors
  (JNIEnv *, jobject, jintArray);

/*
 * Class:     com_itseez_peopledemo_RGBDImage
 * Method:    getHeight
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_itseez_peopledemo_RGBDImage_getHeight
  (JNIEnv *, jobject);

/*
 * Class:     com_itseez_peopledemo_RGBDImage
 * Method:    getWidth
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_itseez_peopledemo_RGBDImage_getWidth
  (JNIEnv *, jobject);

/*
 * Class:     com_itseez_peopledemo_RGBDImage
 * Method:    parse
 * Signature: ([B)V
 */
JNIEXPORT void JNICALL Java_com_itseez_peopledemo_RGBDImage_parse
  (JNIEnv *, jobject, jbyteArray);

#ifdef __cplusplus
}
#endif
#endif
