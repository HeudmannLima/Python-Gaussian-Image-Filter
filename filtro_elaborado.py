import cv2
import numpy as np
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr, shannon_entropy

def filtro(img, peso=0.1, eps=1e-3, num_iter_max=200):

    #alocando memória para iterações
    u = np.zeros_like(img)
    px = np.zeros_like(img)
    py = np.zeros_like(img)
    
    nm = np.prod(img.shape[:2])
    tau = 0.125
    
    i = 0
    while i < num_iter_max:
        u_old = u
        
        #componentes x e y do gradiente de u
        ux = np.roll(u, -1, axis=1) - u
        uy = np.roll(u, -1, axis=0) - u
        
        #atualizando as variaveis px,py
        px_new = px + (tau / peso) * ux
        py_new = py + (tau / peso) * uy
        norm_new = np.maximum(1, np.sqrt(px_new **2 + py_new ** 2))
        px = px_new / norm_new
        py = py_new / norm_new

        #calculando as diferenças
        rx = np.roll(px, 1, axis=1)
        ry = np.roll(py, 1, axis=0)
        div_p = (px - rx) + (py - ry)
        
        #atualizando imagem
        u = img + peso * div_p
        
        #calculando erro
        error = np.linalg.norm(u - u_old) / np.sqrt(nm)
        
        if i == 0:
            err_init = error
            err_prev = error
        else:
            #iteração para se erro for pequeno
            if np.abs(err_prev - error) < eps * err_init:
                break
            else:
                e_prev = error
                
        i += 1
    return u


#definindo as imagens para filtragem
img_ruido = 'questao_noise.png'
img_orign = 'questao_orig.tif'


#armazena imagem e a converte para array
imagem = Image.open(img_ruido)
image1 = misc.fromimage(imagem, flatten = 0)


#executa função filtro() com imagem em array, peso em 46
saida = filtro(image1, peso = 46)


#converte retorno (array) da função p/ imagem 'resultado.png'
misc.imsave('resultado.png', saida)


#armazena imagens originais e resultado em variáveis
original = cv2.imread(img_orign)
ruido_original = cv2.imread(img_ruido)
resultado = cv2.imread('resultado.png')


#converte-as em escala BGR para RGB
orign = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
ruido = cv2.cvtColor(ruido_original, cv2.COLOR_BGR2RGB)
final = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)


#configura-se e exibe plotagem do resultado
fig, ax = plt.subplots(ncols=3, figsize=(11, 5), sharex=True, sharey=True,
subplot_kw={'adjustable': 'box-forced'})

ax[0].imshow(orign)
ax[0].axis('off')
ax[0].set_title('Imagem Original')
ax[1].imshow(ruido)
ax[1].axis('off')
ax[1].set_title('Imagem com Ruído')
ax[2].imshow(final)
ax[2].axis('off')
ax[2].set_title('Imagem com Filtro Aplicado')

fig.tight_layout()
plt.show()


#análise PSNR
p_final = compare_psnr(final, ruido)
p_ruido = compare_psnr(ruido, orign)
p_ro = compare_psnr(final, orign)
print('\n\nPSNR do RESULTADO comparado com RUIDO: ', p_final)
print('PSNR do RUIDO comparado com ORIGINAL: ', p_ruido)
print('PSNR do RESULTADO comparado com ORIGINAL: ', p_ro)


#análise ENTROPY
e_final = shannon_entropy(final)
e_ruido = shannon_entropy(ruido)
e_orign = shannon_entropy(orign)
print('\n\nENTROPY do RESULTADO: ', e_final)
print('ENTROPY do RUIDO: ', e_ruido)
print('ENTROPY do ORIGINAL: ', e_orign)
