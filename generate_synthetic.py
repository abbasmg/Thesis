
# Import packages
from PIL import Image
import numpy as np
from statsmodels.tsa.arima_model import ARIMA


# load the image

def load_img(path=''):
    im = Image.open(path)
    # summarize some details about the image
    print('The image specs are:')
    print(im.format)
    print(im.size)
    im.show()
    gs_image = im.convert(mode='L')
    gs_image = np.array(gs_image)
    gs_image.setflags(write=1)
    gs_image[gs_image<=128] = 1
    gs_image[gs_image>128] = 0
    print('The shape of the data is: '+ str(gs_image.shape))
    return gs_image

path = r'C:\Users\abbme\Desktop\Sine-Wave_1000.jpg'
gs_image=load_img(path=path)

# reduce column dimension
def reduce_coldim(data,mean_len):
    _,b = data.shape
    reduced = []
    for i in range(0, b, mean_len):
       slice_from_index = i
       slice_to_index = slice_from_index + mean_len
       reduced.append(np.mean(data[:,slice_from_index:slice_to_index],axis=1))
    reduced=np.stack(reduced,axis=1)
    return(reduced)  

st = reduce_coldim(gs_image, 8)
print(st.shape)

def gs_to_data(gs):
    _,n = gs.shape
    listofrow = []
    for i in range(n):
      listofrow.append(np.where(gs[:,i]>0))
    data = []
    for i in list(listofrow):
        data.append(np.mean(i))       
    data = np.asarray(data)
    data = data[~np.isnan(data)]
    data = np.interp(data, (data.min(), data.max()), (-1, +1))
    return data

data = gs_to_data(gs_image)

def plot_one_series(data):
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.show()



 
def generate_arima_ts(data, n, seed=12346):
    # model = ARIMA(nts, order=(1,0,2))
    # model_fit = model.fit(disp=0)
    # print(model_fit.summary())
    # plot residual errors
    # residuals = DataFrame(model_fit.resid)
    # residuals.plot()
    # pyplot.show()
    # residuals.plot(kind='kde')
    # pyplot.show()
    # print(residuals.describe())
    # results = model.fit(disp=-1)
    # plt.plot(df_log_shift)
    np.random.seed(seed)
    M = []
    for i in range(n):
      noise = np.random.normal(np.mean(data),0.1,len(data))
      nts = ts + noise
      model = ARIMA(nts,order = (1,0,2))
      m = model.fit(disp = 0)
      M.append(m.fittedvalues)
    M = np.asarray(M)
    return M

generated = generate_arima_ts(data, n= 100)
print(generated.shape)


plot_one_series(generated[69])
