def generate_gif(self, user_id, duration=500):
    image_list = self.get_user_image(user_id)

    # Create a GIF filename based on the current timestamp
    gif_filename = f"{datetime.utcnow()}.gif"

    # Generate the GIF from the image list
    gif_images = []
    for image_name in image_list:
        img = Image.open(image_name)
        gif_images.append(img)

    frames = [Image.fromarray(np.asarray(img)) for img in gif_images]
    frames[0].save(gif_filename, save_all=True, append_images=frames[1:], duration=duration, loop=0)

    # Download the generated GIF
    import requests
    from io import BytesIO

    response = requests.get(gif_filename, stream=True)
    if response.status_code == 200:
        with BytesIO(response.content) as file:
            gif_data = file.read()

        # Download the GIF to the "images" folder in the C drive
        images_folder_path = "C:\\images"
        if not os.path.exists(images_folder_path):
            os.makedirs(images_folder_path)

        gif_download_path = os.path.join(images_folder_path, gif_filename)
        with open(gif_download_path, "wb") as f:
            f.write(gif_data)

        # Delete the temporary GIF file
        os.remove(gif_filename)
