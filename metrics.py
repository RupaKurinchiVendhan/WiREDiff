import numpy as np
import os

MSE = {'WiREDiff': [], 'SR3': [], 'SRCNN': [], 'Bicubic': []}
MAE = {'WiREDiff': [], 'SR3': [], 'SRCNN': [], 'Bicubic': []}

def mse(imageA, imageB):
 # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
 mse_error = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
 mse_error /= float(imageA.shape[0] * imageA.shape[1] * 255 )
 mse_error /= (np.mean((imageA.astype("float"))))**2
 # return the MSE. The lower the error, the more "similar" the two images are.
 return mse_error


def mae(imageA, imageB):
    mae = np.sum(np.absolute((imageB.astype("float") - imageA.astype("float"))))
    mae /= float(imageA.shape[0] * imageA.shape[1] * 255)
    if (mae < 0):
        return mae * -1
    else:
        return mae
    
def compare_output_helper(data_type, component, timestep, i):
    gt_HR = "data/wind_test//HR/{component}_{timestep}_{i}.npy".format(data_type=data_type, component=component, timestep=timestep, i=i)
    wirediff = "wirediff_output/{component}_{timestep}_{i}.npy".format(data_type=data_type, component=component, timestep=timestep, i=i)
    sr3 = "sr3_output/{component}_{timestep}_{i}.npy".format(data_type=data_type, component=component, timestep=timestep, i=i)
    cub = "bicubic/{component}_{timestep}_{i}.npy".format(data_type=data_type, component=component, timestep=timestep, i=i)
    cnn = "cnn/{component}_{timestep}_{i}.npy".format(data_type=data_type, component=component, timestep=timestep, i=i)

    if os.path.isfile(cub) and os.path.isfile(sr3) and os.path.isfile(wirediff):        
        MSE['WiREDiff'].append(mse(np.load(gt_HR), np.load(wirediff)))
        MSE['SR3'].append(mse(np.load(gt_HR), np.load(sr3)))
        MSE['SRCNN'].append(mse(np.load(gt_HR), np.load(cnn)))
        MSE['Bicubic'].append(mse(np.load(gt_HR), np.load(cub)))

        MAE['WiREDiff'].append(mae(np.load(gt_HR), np.load(wirediff)))
        MAE['SR3'].append(mae(np.load(gt_HR), np.load(sr3)))
        MAE['SRCNN'].append(mae(np.load(gt_HR), np.load(cnn)))
        MAE['Bicubic'].append(mae(np.load(gt_HR), np.load(cub)))

if __name__ == '__main__':
    test_wind_timesteps = [2889]
    data_type = 'wind'
    component = None
    for comp in ['ua', 'va']:
            for timestep in test_wind_timesteps:
                for i in range(256):
                    compare_output_helper(data_type, comp, timestep, i)
    for model in MSE.keys():
        print(model)
        print("MSE: ", np.mean(MSE[model]))
        print("MAE: ", np.mean(MAE[model]))
