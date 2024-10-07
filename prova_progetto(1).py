# 1. Installazione e importazione delle librerie

# Installazione della libreria TensorFlow Examples: 
# Questa libreria include vari esempi di modelli di TensorFlow, 
# tra cui pix2pix, che viene utilizzato più avanti nel codice.
pip install git+https://github.com/tensorflow/examples.git


import tensorflow as tf # tensorflow per l'implementazione e l'addestramento del modello.
import tensorflow_datasets as tfds # tensorflow_datasets per caricare set di dati predefiniti
from tensorflow_examples.models.pix2pix import pix2pix # pix2pix per utilizzare i generatori e discriminatori specifici del modello.
# os, time, matplotlib.pyplot, e IPython.display per funzionalità ausiliarie, 
# come la gestione dei file, la visualizzazione delle immagini, e il controllo dell'output.
import os 
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# AUTOTUNE è una funzionalità di TensorFlow che ottimizza automaticamente il prefetching 
# e il caricamento dei dati durante l'addestramento di un modello. 
# Quando lo si utilizza in una pipeline di dati, 
# permette a TensorFlow di determinare dinamicamente il numero ottimale di thread 
# per eseguire operazioni in parallelo, migliorando così le prestazioni della pipeline di dati.
AUTOTUNE = tf.data.AUTOTUNE


# 2. Caricamento del dataset

# tfds.load() carica il dataset, e i dati vengono suddivisi in insiemi di addestramento (trainA, trainB) 
# e di test (testA, testB), rispettivamente per immagini estive e invernali.
# as_supervised=True significa che il dataset verrà caricato come coppie (immagine, etichetta), 
# anche se in questo caso non sono necessarie etichette.
dataset, metadata = tfds.load('cycle_gan/summer2winter',
                              with_info=True, as_supervised=True)

train_summer, train_winter = dataset['trainA'], dataset['trainB']
test_summer, test_winter = dataset['testA'], dataset['testB']



# 3. Configurazione dei parametri
BUFFER_SIZE = 1000 # BUFFER_SIZE: Definisce la dimensione del buffer per lo shuffle del dataset.
BATCH_SIZE = 1 # BATCH_SIZE: Viene impostato a 1 perché CycleGAN utilizza immagini una alla volta durante l'addestramento per ottenere risultati migliori.
IMG_WIDTH = 256 # IMG_WIDTH e IMG_HEIGHT: Le dimensioni dell'immagine sono fissate a 256x256 pixel.
IMG_HEIGHT = 256



# 4. Funzioni di pre-elaborazione delle immagini

# random_crop: Ritaglia l'immagine a una dimensione specifica (256x256). 
# Questa funzione aiuta a introdurre variazioni durante l'addestramento.
def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image




# normalizing the images to [-1, 1]
# normalize: Normalizza l'immagine portando i valori dei pixel nel range [-1, 1], 
# che migliora la stabilità del processo di addestramento.
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image



# random_jitter: Aggiunge disturbi casuali alle immagini per aumentare la robustezza del modello. 
# Esegue i seguenti passaggi:
# Ridimensiona l'immagine a 286x286.
# Ritaglia l'immagine a 256x256.
# Esegue un ribaltamento orizzontale casuale.
def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image



# 5. Applicazione della pre-elaborazione

# preprocess_image_train: Applica sia i disturbi casuali che la normalizzazione per i dati di addestramento.
def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image


# preprocess_image_test: Normalizza solo i dati di test (senza disturbi), poiché i dati di test devono rimanere consistenti.
def preprocess_image_test(image, label):
  image = normalize(image)
  return image



# 6. Configurazione del dataset per l'addestramento

# Caching: Viene utilizzato per memorizzare in cache i dati pre-elaborati per migliorare la velocità.
# map: Applica la funzione di pre-elaborazione al dataset.
# shuffle: Mescola i dati per una migliore convergenza durante l'addestramento.
# batch: Crea batch di dimensione specificata per l'addestramento.
train_summer = train_summer.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_winter = train_winter.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_summer = test_summer.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_winter = test_winter.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)



# 7. Visualizzazione delle immagini

# Estrae immagini di esempio dai set di dati per la visualizzazione e le mostra con o senza disturbi casuali.
sample_summer = next(iter(train_summer))
sample_winter = next(iter(train_winter))


plt.subplot(121)
plt.title('summer')
plt.imshow(sample_summer[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('summer with random jitter')
plt.imshow(random_jitter(sample_summer[0]) * 0.5 + 0.5)


plt.subplot(121)
plt.title('winter')
plt.imshow(sample_winter[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('winter with random jitter')
plt.imshow(random_jitter(sample_winter[0]) * 0.5 + 0.5)



# OUTPUT_CHANNELS indica il numero di canali nell'immagine di output generata dal modello, 
# ed è utilizzato per definire l'architettura del generatore.
# OUTPUT_CHANNELS è impostato a 3, che corrisponde ai tre canali di un'immagine RGB
OUTPUT_CHANNELS = 3

# 8. Costruzione dei Generatori e Discriminatori

# Generatori (generator_g, generator_f): Usano la rete UNet per tradurre le immagini 
# da un dominio all'altro (es: estate -> inverno e viceversa).
generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

# Discriminatori (discriminator_x, discriminator_y): 
# Reti che valutano se un'immagine appartiene al dominio reale o generato.
discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)



# to_winter = generator_g(sample_summer): Questa riga utilizza il generatore generator_g 
# per convertire un'immagine estiva (sample_summer) in un'immagine invernale. 
# Questo generatore è addestrato a mappare il dominio estivo al dominio invernale.
to_winter = generator_g(sample_summer)
# to_summer = generator_f(sample_winter): Questa riga utilizza il generatore generator_f 
# per convertire un'immagine invernale (sample_winter) in un'immagine estiva. 
# Questo generatore è addestrato a fare il percorso inverso, cioè da inverno a estate
to_summer = generator_f(sample_winter)
plt.figure(figsize=(8, 8))
contrast = 8 # contrast = 8: Questo valore viene utilizzato per aumentare il contrasto delle immagini generate, migliorando la visibilità dei dettagli quando si visualizzano le immagini.

imgs = [sample_summer, to_winter, sample_winter, to_summer]
title = ['summer', 'To winter', 'winter', 'To summer']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5) # imgs[i][0] * 0.5 + 0.5: Normalizza i valori dei pixel per riportarli nel range [0, 1] per le immagini non trasformate.
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5) # imgs[i][0] * 0.5 * contrast + 0.5: Aumenta il contrasto delle immagini trasformate utilizzando il valore contrast.
plt.show()




plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real winter?')

# discriminator_y(sample_winter)[0, ..., -1]: Passa l'immagine invernale al discriminatore discriminator_y 
# che valuta quanto l'immagine assomigli a una reale immagine invernale. 
# Viene visualizzata l'ultima mappa di attivazione del discriminatore utilizzando una scala di colori.
plt.imshow(discriminator_y(sample_winter)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real summer?')
# discriminator_x(sample_summer)[0, ..., -1]: Fa lo stesso per il discriminatore discriminator_x, 
# che valuta quanto l'immagine estiva sia reale.
# cmap='RdBu_r': Specifica una mappa di colori per visualizzare la mappa di attivazione del discriminatore. 
# Qui, la scala di colori RdBu_r rappresenta valori positivi e negativi.
plt.imshow(discriminator_x(sample_summer)[0, ..., -1], cmap='RdBu_r')

plt.show()



# LAMBDA: Un parametro che controlla l'importanza della perdita ciclica durante l'addestramento. 
# Viene utilizzato per pesare la perdita di consistenza ciclica rispetto alla perdita avversaria
LAMBDA = 10



# loss_obj: Definisce la funzione di perdita per la classificazione binaria utilizzata dai discriminatori 
# per distinguere tra immagini reali e generate.
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)



# 9. Perdita e Funzioni di Costo

# discriminator_loss: Calcola la perdita per il discriminatore confrontando immagini reali e generate.
# real_loss: La perdita quando il discriminatore classifica correttamente le immagini reali come tali.
# generated_loss: La perdita quando il discriminatore classifica correttamente le immagini generate come false.
def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5



# generator_loss: Calcola la perdita per il generatore cercando di ingannare il discriminatore.
def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)



# calc_cycle_loss: Calcola la perdita di consistenza ciclica che penalizza il generatore 
# se l'immagine tradotta ritorna diversa dall'immagine iniziale.
def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

  return LAMBDA * loss1



# identity_loss: Penalizza il generatore se non riesce a mantenere l'immagine invariata 
# quando l'immagine di input appartiene già al dominio target.
def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss



# Ottimizzatori Adam: Gli ottimizzatori Adam sono utilizzati per aggiornare i pesi dei generatori 
# e dei discriminatori durante l'addestramento. Il parametro beta_1=0.5 aiuta a stabilizzare la convergenza.
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)



# 12. Checkpoint e Salvataggio

# Checkpoint: Il codice salva lo stato del modello durante l'addestramento, permettendo di riprendere da un checkpoint in caso di interruzione.
checkpoint_path = "./checkpoints/train"

# Gestione dei checkpoint: Salva e carica i progressi del modello per prevenire 
# la perdita di dati in caso di interruzione dell'addestramento.
ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

# max_to_keep=5: Limita il numero di checkpoint salvati a 5 per mantenere solo gli ultimi checkpoint.
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')




EPOCHS = 40



# 11. Generazione di Immagini e Valutazione

# generate_images: Visualizza l'immagine di input e la sua versione generata dal modello per valutarne la qualità.
def generate_images(model, test_input):
  prediction = model(test_input)

  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()



# Procedura di Addestramento

# GradientTape: Registra le operazioni per calcolare i gradienti. È persistente perché viene utilizzato più volte.
# Calcola le perdite totali per generatori e discriminatori, compresa la perdita avversaria, ciclica e di identità.
@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)

 # Applicazione dei Gradienti
 # Applica i gradienti calcolati ai rispettivi modelli per aggiornare i pesi e migliorare il modello.
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))



# 10. Addestramento del Modello

# train_step: Funzione che esegue l'addestramento del modello per ogni coppia di immagini di estate e inverno. 
# Effettua il backpropagation e applica i gradienti.
for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_summer, train_winter)):
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  clear_output(wait=True)
  # Using a consistent image (sample_summer) so that the progress of the model
  # is clearly visible.
  generate_images(generator_g, sample_summer)

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))




# 13. Esecuzione finale del modello sui dati di test

# Genera immagini di esempio utilizzando il modello addestrato e le visualizza per valutare le prestazioni.
for inp in test_summer.take(5):
  generate_images(generator_g, inp)




                                                     




