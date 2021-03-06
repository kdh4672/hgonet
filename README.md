## Image-Adaptive Hint Generation via Vision Transformer for Outpainting (WACV 2022 Poster presentation)
```
pip install -r requirements.txt
```


- pre-trained model file
  - https://github.com/kdh4672/hgonet/releases
  - save Inpainting_dis.pth, Inpainting_gen.pth to ```"./checkpoints/vit_side/.```

```
python main.py --path ./checkpoints/vit_side
```


- paper file
  - [Image-Adaptive Hint Generation via Vision Transformer for Outpainting.pdf](https://github.com/kdh4672/hgonet/files/7719685/1196.pdf)

- proposed method
  - ![image](https://user-images.githubusercontent.com/54311546/146195863-ee0880e0-689c-47fd-a2ad-9920c5b2678e.png)
  - ![image](https://user-images.githubusercontent.com/54311546/146195985-a34411ec-8be5-4248-bf1a-2d9570aa3035.png)


