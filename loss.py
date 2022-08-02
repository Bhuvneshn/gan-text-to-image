outputs, activation_real = self.discriminator(right_images, right_embed)
g1_loss = criterion(outputs, smoothed_real_labels) +abs(l1_loss(fake_images, right_images))
g2_loss=criterion(outputs, smoothed_real_labels)+abs(l1_loss(right_images,fake_images))