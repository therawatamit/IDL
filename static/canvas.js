$(document).one('classGiven', function () {
    var canvas = document.getElementById('canvas');
    var bCanvas = document.createElement('canvas');
    var bCtx = bCanvas.getContext('2d');
    var ctx = canvas.getContext('2d');
    $(canvas).css({'max-width': $('#imgInput').width});
    canvas.height = canvas.offsetHeight;
    canvas.width = canvas.offsetWidth;
    var last_mousex = 0;
    var last_mousey = 0;
    var mousex = 0;
    var mousey = 0;
    var mousedown = false;
    var w = canvas.width;
    var h = canvas.height;


    $(canvas).on('mousedown', function (e) {
        last_mousex = parseInt(e.pageX - canvas.offsetLeft);
        last_mousey = parseInt(e.pageY - canvas.offsetTop);
        mousedown = true;
    });


    $(canvas).on('mouseup', function (e) {
        mousedown = false;
    });


    $(canvas).on('mousemove', function (e) {
        mousex = parseInt(e.pageX - canvas.offsetLeft);
        mousey = parseInt(e.pageY - canvas.offsetTop);
        if (mousedown) {
            bCanvas.width = canvas.offsetWidth;
            bCanvas.height = canvas.offsetHeight;
            bCtx.clearRect(0, 0, bCanvas.offsetWidth, bCanvas.offsetHeight); //clear canvas
            bCtx.beginPath();
            var width = mousex - last_mousex;
            var height = mousey - last_mousey;
            bCtx.fillStyle = 'white';
            bCtx.fillRect(last_mousex, last_mousey, width, height);
            bCtx.lineWidth = 1;
            bCtx.stroke();
            ctx.drawImage(bCanvas, 0, 0)
        }
    });

    window.onresize = function () {
        var bcCanvas = document.createElement('canvas');
        bcCanvas.width = canvas.width;
        bcCanvas.height = canvas.height;
        var bcCtx = bcCanvas.getContext('2d');
        bcCtx.drawImage(canvas, 0, 0);
        canvas.height = h;
        canvas.width = w;
        canvas.height = canvas.offsetHeight;
        canvas.width = canvas.offsetWidth;
        ctx.drawImage(bcCanvas, 0, 0, canvas.width, canvas.height);
    }
    $('#clear').click(function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    });
});