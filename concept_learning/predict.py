import numpy as np
import caffe

DEPLOY_PROTOTXT = "vggnet_deploy.prototxt"
TRAINED_NET = "vggnet_vanilla.caffemodel"
IMAGE_PATH = "0002_135225585.jpg"
# instanciate net from saved net
net = caffe.Classifier(DEPLOY_PROTOTXT,TRAINED_NET,caffe.TEST)
caffe.set_mode_cpu()

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['X'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
#transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
#transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
net.blobs['X'].reshape(10,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 224x224
SIZE = 224

# load and transform the image
image = caffe.io.load_image(IMAGE_PATH)
transformed_image = transformer.preprocess('data', image)
# subtract mean

# predict
# copy the image data into the memory allocated for the net
net.blobs['X'].data[...] = transformed_image
### perform classification
output = net.forward()

output_prob_landscape = output['prob_landscape'][0]
output_prob_wildlife = output['prob_wildlife'][0] 
output_prob_travel = output['prob_travel'][0] 
output_prob_vacation = output['prob_vacation'][0] 
output_prob_sunrise = output['prob_sunrise'][0] 
output_prob_sunset = output['prob_sunset'][0] 
output_prob_night = output['prob_night'][0] 
output_prob_art = output['prob_art'][0] 
output_prob_architecture = output['prob_architecture'][0] 
output_prob_urban = output['prob_urban'][0] 
output_prob_abandoned = output['prob_abandoned'][0] 
output_prob_beautiful = output['prob_beautiful'][0] 
output_prob_cute = output['prob_cute'][0] 
output_prob_love = output['prob_love'][0] 
output_prob_beauty = output['prob_beauty'][0]  
output_prob_summer = output['prob_summer'][0]  
output_prob_autumn = output['prob_autumn'][0]  
output_prob_winter = output['prob_winter'][0]  
output_prob_spring = output['prob_spring'][0]  

print "landscape:"
print output_prob_landscape
print "wildlife:"
print output_prob_wildlife
print "travel:"
print output_prob_travel
print "vacation:"
print output_prob_vacation
print "sunrise:"
print output_prob_sunrise
print "sunset:"
print output_prob_sunset
print "night:"
print output_prob_night
print "art:"
print output_prob_art
print "architecture:"
print output_prob_architecture
print "urban:"
print output_prob_urban
print "abandoned:"
print output_prob_abandoned
print "beautiful:"
print output_prob_beautiful
print "cute:"
print output_prob_cute
print "love:"
print output_prob_love
print "beauty:"
print output_prob_beauty
print "summer:"
print output_prob_summer
print "autumn:"
print output_prob_autumn
print "winter:"
print output_prob_winter
print "spring:"
print output_prob_spring
