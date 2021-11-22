<html lang="en"><head>
    <meta charset="UTF-8">
    <title></title>
<style id="system" type="text/css">h1,h2,h3,h4,h5,h6,p,blockquote {    margin: 0;    padding: 0;}body {    font-family: "Helvetica Neue", Helvetica, "Hiragino Sans GB", Arial, sans-serif;    font-size: 13px;    line-height: 18px;    color: #737373;    margin: 10px 13px 10px 13px;}a {    color: #0069d6;}a:hover {    color: #0050a3;    text-decoration: none;}a img {    border: none;}p {    margin-bottom: 9px;}h1,h2,h3,h4,h5,h6 {    color: #404040;    line-height: 36px;}h1 {    margin-bottom: 18px;    font-size: 30px;}h2 {    font-size: 24px;}h3 {    font-size: 18px;}h4 {    font-size: 16px;}h5 {    font-size: 14px;}h6 {    font-size: 13px;}hr {    margin: 0 0 19px;    border: 0;    border-bottom: 1px solid #ccc;}blockquote {    padding: 13px 13px 21px 15px;    margin-bottom: 18px;    font-family:georgia,serif;    font-style: italic;}blockquote:before {    content:"C";    font-size:40px;    margin-left:-10px;    font-family:georgia,serif;    color:#eee;}blockquote p {    font-size: 14px;    font-weight: 300;    line-height: 18px;    margin-bottom: 0;    font-style: italic;}code, pre {    font-family: Monaco, Andale Mono, Courier New, monospace;}code {    background-color: #fee9cc;    color: rgba(0, 0, 0, 0.75);    padding: 1px 3px;    font-size: 12px;    -webkit-border-radius: 3px;    -moz-border-radius: 3px;    border-radius: 3px;}pre {    display: block;    padding: 14px;    margin: 0 0 18px;    line-height: 16px;    font-size: 11px;    border: 1px solid #d9d9d9;    white-space: pre-wrap;    word-wrap: break-word;}pre code {    background-color: #fff;    color:#737373;    font-size: 11px;    padding: 0;}@media screen and (min-width: 768px) {    body {        width: 748px;        margin:10px auto;    }}</style><style id="custom" type="text/css"></style></head>
<body marginheight="0"><h2>The code for DBIA on DeiT</h2>
<ul>
<li><p>DeiT's code reference is from <a href="https://github.com/facebookresearch/deit">https://github.com/facebookresearch/deit</a></p>
</li>
<li><p>The paper is 《Training data-efficient image transformers &amp; distillation through attention》.</p>
</li>
</ul>
<h3>Environment</h3>
<ul>
<li>PyTorch 1.7.0+ and torchvision 0.8.1+ and pytorch-image-models 0.3.2</li>
</ul>
<h3>Data Preparation</h3>
<ul>
<li>Imagenet2012 validation set data is placed in <code>./data/imagenet2012/val</code></li>
</ul>
<p><img src="" alt="">

</p>
<h3>Starting Poisoning</h3>
<ul>
<li>Generate the background picture required for training trigger<pre><code>  python preprocess.py</code></pre>
</li>
<li>Background picture and <code>blend_ tensor.pt</code> is stored in <code>./savedfigure</code></li>
</ul>
<h3>Reversing Trigger</h3>
<ul>
<li>First, write the path of <code>blend_tensor.pt</code> on line 38 of the file <code>activate_trigger.py</code><pre><code>  python activate_trigger.py</code></pre>
</li>
<li>Trigger pictures are stored in the triggers folder</li>
</ul>
<h3>Training Poisoning Model</h3>
<ul>
<li>First, write the path of the selected trigger on line 35 of the file <code>tensors_dataset.py</code><pre><code>  python backdoor_main.py</code></pre>
</li>
</ul>
</body></html>