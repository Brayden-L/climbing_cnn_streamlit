import streamlit as st
import os
from utility_funcs import *
from PIL import Image, ImageOps
from io import BytesIO

# Get list of pre-provided images
included_imgs = list_images_in_folder(r'./images')

# Initialize session state variables
if 'img_index' not in st.session_state:
    st.session_state.img_index = 0

# Streamlit Init
st.set_page_config(layout="wide", page_title="Main Page")
st.title('Climbing Photo CNN')
with st.expander('What Is This?'):
    st.markdown(help_text)

col1, col2, col3 = st.columns([1,1,1])

# Data Input options
with col1:
    model_selection = st.radio(label='Model Type', options=['VGG11', 'AlexNet'], help='VGG11 is a more accurate model, AlexNet is a simpler model.')
    st.header('Try an Example Photo')
    ex_butt = st.button(f'See Another Example Photo')

    st.header('Upload Your Own Photo')
    uploaded_file = st.file_uploader(label='Upload', accept_multiple_files=False, help='No grayscale photos, RGB only! .JPG preferred.')

# Model run and display results function
def run_and_display_model(image_to_use, caption):
    display_image = Image.open(image_to_use)
    display_image = resize_image_object_to_height(display_image, 400)
    
    with col2:
        st.image(display_image, caption='')
        st.write(caption)
    
    with col3:
        with st.spinner('Running Model'):
            # Transform the image for the model
            image_for_model = transform_image(image_to_use)

            model = load_retrain_model(model_selection)    
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                probabilities, prob_list_raw = predict_from_model(model, image_for_model)
                prob_df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability']).set_index('Class')
                prob_df.index.name = 'Class'

                st.subheader('Class Probabilities', help='This is the probability that the image belongs to a given class according to the model.')
                st.table(prob_df)
                
                if all(p < 0.9 for p in prob_list_raw):
                    st.warning("Warning: \n\n The model is not > 90% certain of any given class. \n\n The image is potentially borderline or does not belong to any of the classes.")


# Model run logic

# If user has not yet interacted with the page, give an example.
if uploaded_file is None and ex_butt is False:
    ex_img = included_imgs[st.session_state.img_index]
    img_caption = img_captions[os.path.basename(ex_img)]
    run_and_display_model(ex_img, img_caption)

# If user requests an example, load the next one, and remove the uploaded photo if one has been uploaded.
if ex_butt:
    st.session_state.img_index = (st.session_state.img_index + 1) % len(included_imgs)  # Increment and loop back if at the end
    uploaded_file = None
    ex_img = included_imgs[st.session_state.img_index]
    img_caption = img_captions[os.path.basename(ex_img)]
    run_and_display_model(ex_img, img_caption)

if uploaded_file is not None:
    # Rotate the file if it has exif data
    uploaded_file_rot = Image.open(uploaded_file)
    if uploaded_file_rot._getexif():
        uploaded_file_rot = ImageOps.exif_transpose(uploaded_file_rot)
    else:
        pass
    
    # Re-save file to bytestream, which is what the streamlit file-upload type is
    uploaded_file_rot_bytestream = BytesIO()
    uploaded_file_rot.save(uploaded_file_rot_bytestream, format='JPEG')
    
    img_caption = 'Uploaded Image'
    run_and_display_model(uploaded_file_rot_bytestream, img_caption)