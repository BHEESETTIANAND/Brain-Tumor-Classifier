# import libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv
import openai
import io
import base64

# Configuration
load_dotenv()

# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
# openai.api_key = os.getenv('OPENAI_API_KEY')
# client = openai.OpenAI()

def plot_predictions(predictions, class_index, predicted_label):
 
    predictions = [p * 100 for p in predictions]
    
    # Create the bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=predictions,               # Set x to predictions for horizontal bars
            y=sorted_labels,                    # Set y to labels
            orientation='h',             # 'h' for horizontal orientation
            text=[f"{p:.2f}%" for p in predictions],
            textposition="auto",
            marker=dict(color=['#32CD32', '#FFD700', '#FF4500', '#B22222']),
            opacity=0.7
        )
    ])

    # Customize layout
    fig.update_layout(
        title="Model Predictions",
        title_font=dict(size=25, family="Arial"),
        xaxis_title="Prediction Probability",
        yaxis_title="Tumor Type",
        xaxis=dict(range=[0, 101]),
        template="plotly_white"
    )

    return fig

output_dir = 'saliency_maps'
os.makedirs(output_dir, exist_ok=True)

def generate_saliency_map(model, img_array, class_index, img_size, file, filename):
  # Set up a tape to watch gradients while predicting
  with tf.GradientTape() as tape:
    # converts img array to tensor to be processed for tf
    img_tensor = tf.convert_to_tensor(img_array)
    # watches gradients
    tape.watch(img_tensor)
    # extracts predictions of model
    predictions = model(img_tensor)
    # probability of target class
    target_class = predictions[:, class_index]

  # run one more "backward propagation" to extract gradients with respect to the target class
  gradients = tape.gradient(target_class, img_tensor)
  # extract absolute values since we only care about how the magnitute of importance not whether it increased or decreased
  gradients = tf.math.abs(gradients)
  # extracts only the strongest gradient out of the three channels RGB -> 1 gradient per pixel 
  gradients = tf.reduce_max(gradients, axis=-1)
  # sequeezes out singleton dimensions of size 1
  gradients = gradients.numpy().squeeze()

  # Resize gradients to match original image size
  gradients = cv2.resize(gradients, img_size)

  ## Create a circular mask for the brain area to focus on the brain and ignore dark background

  center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
  # -10 to leave a small margin for radius
  radius = min(center[0], center[1]) - 10
  # creates a grid for the pixels based on gradients size
  y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
  # equation of a circle (x-h)^2 + (y-k)^2 <= r^2
  # mask returns a grid of 1/0 of whether a pixel is in the circular center
  mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
  # Apply mask to gradients
  gradients = gradients * mask

  ## Normalize only the brain area

  # extract pixels only in the mask
  brain_gradients = gradients[mask]
  # check if it is already uniform
  if brain_gradients.max() > brain_gradients.min():
    brain_gradients = (brain_gradients - brain_gradients.min()) / (brain_gradients.max() - brain_gradients.min())
  # update normalized gradients only in the mask area
  gradients[mask] = brain_gradients

  # Apply a higher threshold of 80%
  threshold = np.percentile(gradients[mask], 80)
  # only keep the top 20% strongest gradients
  gradients[gradients < threshold] = 0

  # Apply more aggressive smoothing
  gradients = cv2.GaussianBlur(gradients, (11,11), 0)

  # Create a heatmap overlay with enhanced contrast
  heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

  # Resize heatmap to match original image size
  heatmap = cv2.resize(heatmap, img_size)

  ## Superimpose the heatmap on original image with increased opacity
  original_img = image.img_to_array(img)
  # 70% opacity for heatmap on top of 30% of the MRI sacn
  superimposed_img = heatmap * 0.7 + original_img * 0.3
  superimposed_img = superimposed_img.astype(np.uint8)

  # get MRI image path
  img_path = os.path.join(output_dir, filename)
  # save MRI image
  with open(img_path, "wb") as f:
    f.write(file.getbuffer())

  # define map path
  saliency_map_path = f'saliency map/{filename}'

  # Save saliency map
  cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

  return superimposed_img

def load_transfer_model(model_path, model_name):
  
  if model_name == 'xception':
    img_shape = (299,299, 3)
    base_model = tf.keras.applications.xception.Xception(include_top=False,
                                                        weights='imagenet',
                                                        input_shape=img_shape,
                                                        pooling='max')
  elif model_name == 'mobilenet':
    img_shape = (224,224, 3)
    base_model = tf.keras.applications.MobileNet(include_top=False,
                                                       weights='imagenet',
                                                       input_shape=img_shape,
                                                       pooling='max')
  
  model = Sequential([
      base_model,
      Flatten(),
      Dropout(rate=0.3),
      Dense(128, activation='relu'),
      Dropout(rate=0.25),
      Dense(4, activation='softmax')
  ])

  model.build((None,) + img_shape)

  model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', Precision(), Recall()])

  model.load_weights(model_path)

  return model


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
    
st.title('Brain Tumor Classifier')
                
image_paths = ["Select an image"] + [os.path.join('images', file) for file in os.listdir('images') if os.path.isfile(os.path.join('images', file))]

col1, col2 = st.columns(2)
with col1:
    selected_image = st.selectbox("Select an image from dataset", image_paths)
    selected_file = None    
    if selected_image != 'Select an image':
        selected_file = open(selected_image, "rb")  # Create a file variable based on selected_image
        selected_file = selected_file.read()  # Read the file content into a variable
        selected_file = io.BytesIO(selected_file)  # Convert to BytesIO for compatibility
        selected_file_name = os.path.basename(selected_image)
        source = "select"
        uploaded_file = None
        base64_image = encode_image(selected_image)
        
with col2:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        source = "upload"
        selected_file = None
        selected_image = "Select an image"

if 'selected_image' not in st.session_state:
    st.session_state.selected_image = "Select an image"

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if uploaded_file is not None or selected_file is not None:

    print('uploaded file', uploaded_file)
    print('st session uploaded file', st.session_state.uploaded_file)
    
    print('selected file', selected_image)
    print('st session selected file', st.session_state.selected_image)
    if uploaded_file != st.session_state.uploaded_file or selected_image != st.session_state.selected_image:
        st.session_state.initialized = False

    print('init', st.session_state.initialized)
    st.session_state.uploaded_file = uploaded_file
    st.session_state.selected_image = selected_image

    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = ""

    selected_model = st.selectbox(
    "Select Model:",
    ("Xception - Transfer Learning", "MobileNet - Transfer Learning", "Custom CNN"))

    
    if selected_model != st.session_state.selected_model:
        st.session_state.initialized = False
    
    st.session_state.selected_model = selected_model

    if selected_model == 'MobileNet - Transfer Learning':
        model = load_transfer_model('models/MobileNet_model.weights.h5', 'mobilenet')
        img_size = (224,224)
    elif selected_model == "Xception - Transfer Learning":
        model = load_transfer_model('models/xception_model.weights.h5', 'xception')
        img_size =(299,299)
    else:
        model = load_model('models/cnn_model.h5')
        img_size = (224,224)

    if source == 'upload':
        img = image.load_img(uploaded_file, target_size=img_size)
    elif source == 'select':
        img = image.load_img(selected_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)


    labels = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
    sorted_labels = ['No tumor', 'Pituitary', 'Meningioma','Glioma']

    label_index_map = {label: index for index, label in enumerate(sorted_labels)}

    sorted_predictions = [predictions[0][labels.index(label)] for label in sorted_labels]

    class_index = np.argmax(sorted_predictions)

    result = sorted_labels[class_index]

    color_map = {
    "Glioma": "#B22222",
    "Meningioma": "#FF4500",
    "No tumor": "#32CD32",
    "Pituitary": "#FFD700"
    }

    st.write("### Classification Results")
    result_container = st.container()
    result_container.markdown(
    f"""
    <div style="background-color: #000000; color: #ffffff; padding: 30px; border-radius: 15px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="flex: 1; text-align: center;">
            <h3 style="color: #ffffff; margin-bottom: 10p; font-size: 20px;">Predicted Class</h3>
            <p style="font-size: 36px; font-weight: 800; color: {color_map[result]}; margin: 0;">
                {result}
            </p>
        </div>
        <div style="width: 2px; height: 80px; background-color: #ffffff; margin: 0 20px;"></div>
        <div style="flex: 1; text-align:center;">
            <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px">Confidence</h3>
            <p style="font-size: 36px; font-weight: 800; color: #2196F3; margin: 0;">
                {sorted_predictions[class_index]:.4%}
            </p>
        </div>
    </div>
    </div>
    """,
    unsafe_allow_html=True
    )


    fig = plot_predictions(sorted_predictions, class_index, result)
    st.plotly_chart(fig, use_column_width=True)

    st.write("### Saliency Map")
    if source == 'upload':
        saliency_map = generate_saliency_map(model, img_array, class_index, img_size, uploaded_file, uploaded_file.name)
    elif source == 'select':
        saliency_map = generate_saliency_map(model, img_array, class_index, img_size, selected_file, selected_file_name)

    col1, col2 = st.columns(2)
    with col1:
        if source == 'upload':
            st.image(uploaded_file, caption="Uploaded MRI Scan", use_column_width=True)
        elif source == 'select':
            st.image(selected_file, caption="Uploaded MRI Scan", use_column_width=True)
    with col2:
        st.image(saliency_map, caption="Saliency Map", use_column_width=True)

    if source == 'upload':
        saliency_map_path = f'saliency_maps/{uploaded_file.name}'
    elif source == 'select':
        saliency_map_path = f'saliency_maps/{selected_file_name}'

    confidence = sorted_predictions[class_index]

    # st.write("## Explanation")

    # if 'initialized' not in st.session_state:
    #     st.session_state.initialized = False

    # if 'user_prompt' not in st.session_state:
    #     st.session_state.user_prompt = ""

    # user_prompt = st.text_input("How would you like the results to be explained ?", placeholder="Explain to me like I am a 5 year old")

    # if user_prompt != st.session_state.user_prompt:
    #     st.session_state.initialized = False
    
    # st.session_state.user_prompt = user_prompt

    # image_prompt = f"""
    # You are an expert neurologist tasked with explaining a brain tumor MRI scan. The scan includes a visual map that highlights regions of interest.

    # The highlighted regions, especially the light cyan areas, represent the strongest gradients in the analysis. These regions indicate where the model focused most heavily when determining the classification.

    # The analysis has classified the MRI as {result}. This is the user's prompt on how they want the results to be explained: {st.session_state.user_prompt}.

    # Discuss the highlighted map:

    # Explain the specific regions of the brain the cyan areas correspond to. Mention any critical regions contributing to the classification.
    # Identify the anatomical or functional brain areas, such as the frontal lobe, temporal lobe, or brainstem, emphasized in the map, if relevant.
    # Interpret the analysis by explaining how these highlighted regions might relate to the characteristics of {result}, such as tumor size, texture, or location.
    # Let's think about this step by step to ensure clarity.

    # NEVER mention the saliency map, machine learning, or the confidence of the prediction. Limit your explanation to 5 sentences maximum. 
    # """

    # chat_prompt = f"""
    # You are an expert neurologist and radiologist, with extensive experience explaining medical imaging and diagnostic insights in a clear, patient-centered manner. Your task is to discuss the results of a brain MRI scan, focusing on the highlighted regions that indicate critical areas of interest, and the classification of a potential tumor.

    # The diagnostic analysis categorizes the tumor into one of the following: glioma, meningioma, pituitary tumor, or no tumor.

    # The highlighted regions, particularly those in light cyan, represent areas of significant focus. These regions correspond to key areas that influenced the classification based on the scan’s features.

    # The analysis concluded that the tumor is classified as {result}.

    # Here’s how you should guide the conversation:

    # Discuss the highlighted map:

    # Explain which parts of the brain are emphasized and why they are critical for the classification.
    # Highlight any specific anatomical or functional brain regions, such as the frontal lobe, temporal lobe, or brainstem, that correspond to the areas of interest.
    # Interpret the classification:

    # Describe the reasoning behind the classification of {result}, step by step, based on the features and characteristics of the tumor type.
    # Mention factors that may have contributed to this conclusion, such as the size, texture, or location of the tumor, as seen in the scan.
    # Adapt to the user’s knowledge level:

    # For users unfamiliar with medical details, provide a simplified explanation. For example, describe the cyan regions as showing the parts of the brain where the scan revealed the most notable differences indicative of a tumor.
    # If the user seeks more detail, delve into advanced aspects, such as how specific brain structures correlate with the tumor type and the general diagnostic process.
    # Enable a continued conversation:

    # Invite the user to ask follow-up questions to clarify or expand on the explanation.
    # Adjust your response dynamically based on the user’s level of understanding or curiosity, using the style indicated in their prompt: {st.session_state.user_prompt}.
    # IMPORTANT:

    # Do not mention the terms "saliency map" or "machine learning model."
    # Tailor the explanation to ensure it aligns with the user’s preferred style and understanding.
    # Keep the response conversational and patient-focused, encouraging an ongoing dialogue.
    # """

    # if user_prompt:
    #     response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[
    #         {
    #         "role": "system",
    #         "content": image_prompt
    #         },
    #         {
    #         "role": "user", "content": 
    #         [
    #             {"type": "text", "text": user_prompt},
    #             {"type": "image_url", "image_url": 
    #                 {"url": f"data:image/jpeg;base64,{base64_image}"} 
    #             }
    #         ]
    #         }
    #     ],
    #     )

    #     if not st.session_state.initialized:
    #         st.session_state.messages = []
    #         st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
    #         st.session_state.initialized = True


    # if 'messages' in st.session_state:
    #     for message in st.session_state.messages:
    #         with st.chat_message(message['role']):
    #             st.markdown(message['content'])

    # # Accept user input
    # if user_prompt:
    #     if user_input:= st.chat_input(""):
    #         # Add user message to chat history
    #         st.session_state.messages.append({"role": "user", "content": user_input})
    #         with st.chat_message('user'):
    #             st.markdown(user_input)
    #         with st.chat_message("assistant"):
    #             stream = client.chat.completions.create(
    #                 model="gpt-4o-mini",
    #                 messages=[
    #                     {"role": "system", "content": chat_prompt},
    #                     {"role": "user", "content": 
    #                         [
    #                             {"type": "text", "text": user_prompt},
    #                             {"type": "image_url", "image_url": 
    #                                 {"url": f"data:image/jpeg;base64,{base64_image}"} 
    #                             }
    #                         ]
    #                     },
    #                     *[
    #                         {"role": m["role"], "content": m["content"]}
    #                         for m in st.session_state.messages
    #                     ]
    #                 ],
    #                 stream=True,    
    #             )
    #             response = st.write_stream(stream)

    #         st.session_state.messages.append({"role": "assistant", "content": response})
    #         print(response)
