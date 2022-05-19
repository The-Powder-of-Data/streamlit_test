def label_to_rgb(predicted_image):
    """
    converts predicted mask into label rgb colors
    """
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
    
    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    Road = '#6EC1E4'.lstrip('#') 
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
    
    Vegetation =  'FEDD3A'.lstrip('#') 
    Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58
    
    Water = 'E2A929'.lstrip('#') 
    Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41
    
    Unlabeled = '#9B9B9B'.lstrip('#') 
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155
    
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    
    segmented_img[(predicted_image == 0)] = Building
    segmented_img[(predicted_image == 1)] = Land
    segmented_img[(predicted_image == 2)] = Road
    segmented_img[(predicted_image == 3)] = Vegetation
    segmented_img[(predicted_image == 4)] = Water
    segmented_img[(predicted_image == 5)] = Unlabeled
    
    segmented_img = segmented_img.astype(np.uint8)
    return(segmented_img)

def pipeline_pred_smooth(image_location):
  """
  Takes an image directory and produces a segmentation mask using a smoothed edge tile approach

  0. input directory for image
  1. image scaled
  2. break image into tiles w overlapping pixels
  3. use model to predict segemntation on each tile
  4. convert array back to rgb
  5. display
  """
  img = cv2.imread(image_location)
  input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

  predictions_smooth = predict_img_with_smooth_windowing(
      input_img,
      window_size=patch_size,
      subdivisions=2,  
      nb_classes=n_classes,
      pred_func=(
          lambda img_batch_subdiv: model.predict((img_batch_subdiv))))

  final_prediction = np.argmax(predictions_smooth, axis=2)
  prediction_w_smooth = label_to_rgb(final_prediction) 

  plt.figure(figsize=(16, 16))
  plt.subplot(221)
  plt.title('Testing Image')
  plt.imshow(img)
  plt.axis('off')

  plt.subplot(222)
  plt.title('Prediction with smooth blending')
  plt.imshow(prediction_w_smooth)
  plt.axis('off')