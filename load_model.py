#!/usr/bin/env python3

import numpy as np
from tensorflow.keras.models import load_model
from deconv_ae import slice_images

from astropy.io import fits


def combine_blocks_3d(input_block,output_shape=(901,1024,1024),overlap=0,pad_x=0,pad_y=0,weight=None):
    """Combine blocks into a single image"""
    dd,rr,cc = output_shape
    fsz,bsz,ssz = input_block.shape
    image = np.zeros(output_shape)
    block_counter = 0
    blocks_per_freq = int((np.ceil((rr - bsz) / (bsz - overlap) + 1)) * (np.ceil((cc - bsz) / (bsz - overlap)) + 1))
    for d in range(dd):
        depth_blocks = input_block[block_counter:block_counter+blocks_per_freq]
#        print(depth_blocks.shape,bsz,ssz)
#        depth_blocks = depth_blocks.reshape((-1,rr,cc))
#        print(depth_blocks.shape)
#        assert depth_blocks.shape[0] == 1
#        depth_blocks = depth_blocks[0]
        image[d] = combine_blocks(depth_blocks,(rr,cc),overlap,pad_x,pad_y,weight)
#        image[d] = depth_blocks
        block_counter += blocks_per_freq
    return image

def create_linear_weight(msz,csz):
    outvalue = 0.01
    weight = np.ones([msz,msz])*outvalue
    center_val = 1
    decrement = (center_val - outvalue) / ((msz - csz) // 2 )
    for i in range(0, (msz-csz)//2):
        weight[i:-i,i:-i] = outvalue + decrement * i

    center_start = (msz - csz) // 2
    center_end = center_start + csz
    weight[center_start:center_end,center_start:center_end] = center_val

    return np.array(weight)


def combine_blocks(input_block,output_shape=(1024,1024),overlap=0,pad_x=0,pad_y=0,weight=None):
    rr,cc = output_shape
    image = np.zeros((output_shape[0]+pad_x,output_shape[1]+pad_y))
    num,bsz,ssz = input_block.shape
    assert bsz == ssz
    block_count = np.zeros(np.shape(image))
    stride = np.array([bsz - overlap, ssz - overlap])
    blocke_counter = 0


    for i in range(0,rr-bsz+pad_x+1,stride[0]):
        for j in range(0,cc-ssz+pad_y+1,stride[1]):
            image[i:i+bsz,j:j+bsz] += input_block[blocke_counter]*weight
            block_count[i:i+bsz,j:j+bsz] += weight
            blocke_counter += 1

    image /= block_count
    image = image[0:rr,0:cc]

    return image

def normalize_image(image):
    a = np.mean(image)
    b = np.std(image)
    image = (image - a) / b
    return image,a,b

def apply_normalization(image,mean,std):
    image = ( image - mean ) / std
    return image

def inverse_normalize_image(image,mean,std):
    image = image * std + mean
    return image

def main(model_path,infile,psf_file,outfile):
#    model = load_model(model_path)
    header = fits.getheader(infile)
    weight=create_linear_weight(128,100)
    data = fits.getdata(infile)
    if psf_file is not None:
        psf = fits.getdata(psf_file)
    mean = np.mean(data)
    std = np.std(data)
    freq_array = np.linspace(60,70.22,512)
    print(mean,std)
    Flag_test = False
    Flag_simple = 1000
    Flag_model = 1
    Flag_norm = True
    if Flag_test:
        data=data[0:10]
        blocks,pad_x,pad_y = slice_images(data,128,100)
#    print(pad_x,pad_y)
        predicted_image = combine_blocks_3d(blocks,output_shape=data.shape,overlap=28,pad_x=pad_x,pad_y=pad_y,weight=weight)
        data = apply_normalization(data,mean,std)
    elif len(data.shape) == 3:
        model = load_model(model_path)
        if Flag_norm:
            data = apply_normalization(data,mean,std)
        data_predict = np.zeros(data.shape)
        for i in range(len(data)):
            data_use = data[i].reshape(-1,1800,1800)
            sliced_data,pad_x,pad_y = slice_images(data_use,128,100)
            if Flag_model == 1:
                blocks = sliced_data
            else:
                psf_use = (psf[i] * freq_array[i]).reshape(-1,128,128)
                blocks = np.zeros((len(sliced_data),128,128,2))
                for j in range(len(sliced_data)):
                    blocks[j,:,:,0] = sliced_data[j]
                    blocks[j,:,:,1] = psf_use
            predicted_blocks = np.squeeze(model.predict(blocks))
            predicted_image = combine_blocks_3d(predicted_blocks,output_shape=data_use.shape,overlap=28,pad_x=pad_x,pad_y=pad_y,weight=weight)
            if Flag_norm:
                predicted_image = inverse_normalize_image(predicted_image,mean,std)
            data_predict[i] = predicted_image
            if i >= Flag_simple:
                break
    fits.writeto(outfile,data_predict,header=header, overwrite=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Load a trained model and predict on an image')
    parser.add_argument('-m',dest='model_path', type=str, help='Path to trained model')
    parser.add_argument('-i',dest='infile', type=str, help='Path to input image')
    parser.add_argument('-o',dest='outfile', type=str, help='Path to output image')
    parser.add_argument('-p',dest='psf', type=str, help='Path to psf image',default=None)
    args = parser.parse_args()
    main(args.model_path,args.infile,args.psf, args.outfile)

