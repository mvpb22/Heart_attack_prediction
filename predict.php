  <!DOCTYPE html>
  <html lang="en" dir="ltr">
    <head>
       <title>Heart Disease Prediction</title>
       <meta charset="utf-8">
       <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css"
       integrity="sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO" crossorigin="anonymous">
       <link rel="stylesheet" type="text/css" href="/css/global.css">
       <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='css/Style.css')}}">

       <style>
           body{
               #background: #e6ccff;
               background: linear-gradient(to left, #667292 0%, #a2b9bc 100%);
           }
           footer a{
             color: #fff;
           }
           footer a:hover{
             color: #fff;
           }
           .centerdiv{
             height: 15vh;
             display: flex;
             #justify-content: center;
             align-items: center;
           }
           .centerdiv a{
             height: 30px;
             width: 30px;
             background-color: #f5f6fa;
             border-radius: 50px;
             text-align: center;
             margin: 5px;
             line-height: 30px;
             #box-shadow: 1px 4px 2px 2px #dcdde1;
             position: relative;
             overflow: hidden;
           }
           .centerdiv a i{
             transition: all 0.3s linear;
           }
           .centerdiv a:hover i{
             transform: scale(1.5);
             color: #f5f6fa;
           }
           .centerdiv a:before{
             content: "";
             width: 120%; height: 120%;
             position: absolute;
             top: 90%; left: -50%;
             background-color: #00a8ff;
             transform: rotate(60deg);
           }

           .centerdiv a:hover:before{
             animation: socialicons 0.8s 1;
             animation-fill-mode: forwards;
           }
           @keyframes socialicons {

             0%{ top: 90%; left: -50%;}
             50%{ top: -60%; left: -10%;}
             100%{ top: -10%; left: -10%}
           }

           .fa-facebook-f{
             color: #e84393;
           }
           .fa-instagram{
             color: #e84393;
           }
           .fa-github{
             color: #e84118;
           }
           .fa-linkedin{
             color: #0097e6;
           }
           .fa-twitter{
             color: #0097e6;
           }
       </style>

    </head>

  <body>



  <h1>
    <div style ="text-align:center">
      <font color='white'>
        Heart Attack Risk Prediction
        <center><h5>Disclaimer: The output predicted will only based on the model. However, it might be better to talk to a doctor regardless.</h5></center>
      </font>
    </div>
  </h1>
  <br>


   <section id="facilities">
     <div class="container">
       <div class="card" style= "max-width: 500px; margin:0 auto;">
         <div class="card-body">
       <div class="title">
         <h3>
           <font color='Black'>Our model predicted that you might be </font>
         </h3>
         <font color='white'>
           <h1>
             <font color="blue">{{ prediction}}</font>
          </h1>
         </font>
         <h3>
           <font color='Black'>by Heart Attack</font>
         </h3>
       </div>
     </div>
     </div>
     </div>
   </section>


  <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"
  integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js"
  integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
  <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/js/bootstrap.min.js"
  integrity="sha384-ChfqqxuZUCnJSK3+MXmPNIyE6ZbWh2IMqE241rYiqJxyMiZ6OW/JmZQ5stwEULTy" crossorigin="anonymous"></script>
  </body>
  </html>
