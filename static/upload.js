$(function () {
    function showInput(input) {
        if (!input.files || !input.files[0]) {
            return;
        }
        var w = 0, h = 0
        var reader = new FileReader();
        reader.onload = function (e) {
            var image = new Image();
            image.src = e.target.result;
            image.onload = function () {
                w = this.width;
                h = this.height;
                var ch=0;
                let x = window.location.href.substring(window.location.href.lastIndexOf('/') + 1).toString();
                if (x==""||x=="denoise"){
                    ch = 1000;
                }
                else if(x=="superresolution"){
                    ch = 400;
                }
                else if(x=="arti"){
                    ch = 2000;
                }
                else if(x=="inpaint"){
                    ch = 756;
                }
                $('.grid-input').removeClass('d-none');
                if (w * h <= ch * ch) {
                    $('#imgInput').attr('src', e.target.result);
                    $('.btns').removeClass('d-none');
                    $('#imgInput').css({'border-width': '0', 'min-height': '0'});
                    $('.grid-output iframe').remove();
                    $(document).trigger('classGiven');
                    var c = document.getElementById('canvas')
                    $(c).css({'width': w.toString(), 'height': h.toString()});
                    $(window).trigger('resize');
                } else {
                    $('#imgInput').attr('alt', "Please select an image with dimensions less than or equivalent to "+ch+"x"+ch);
                    $('#imgInput').attr('src', "");
                    $('.btns').addClass('d-none');
                    input.files[0] = null;

                }

            }
        }
        reader.readAsDataURL(input.files[0]);
        $('.grid-show p').remove();
    }

    function sendFile() {
        let form = document.getElementById("imgForm")
        let formData = new FormData(form);
        $('.loading').removeClass('d-none');
        if (!$('.grid-output').hasClass('d-none')) {
            $('.grid-output').addClass('d-none');
        }
        if ($('#canvas').length > 0) {
            var dataURL = document.getElementById('canvas').toDataURL();
            var blobBin = atob(dataURL.split(',')[1]);
            var blobarray = [];
            for (var i = 0; i < blobBin.length; i++) {
                blobarray.push(blobBin.charCodeAt(i));
            }
            var mask = new Blob([new Uint8Array(blobarray)], {type: 'image/png'});
            formData.append("mask", mask);
        }
        $.ajax({
            url: form.getAttribute('action'),
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
                showOutput(data);
                $('.loading').addClass('d-none');
                $('.grid-output').removeClass('d-none');
            },
            error: function (xhr, err) {
                console.error(err);
                $('#imgOutput').attr('src', '');
                $('.loading').addClass('d-none');
                var frame = document.createElement('iframe');
                frame.addEventListener('load', function () {
                    var b = $('.grid-show iframe').contents().find('body');
                    b.append(xhr.responseText);
                    console.log(xhr.responseText.toString() + '\n');
                }, false);
                $('.grid-show')[0].append(frame);

            }
        });
    }

    function showOutput(data) {
        $('#imgOutput').attr('src', 'data:image/jpeg;base64,' + data);
        $('a.dl').attr('href', 'data:image/jpeg;base64,' + data);
    }

    $("#file").change(function () {
        showInput($("#file")[0]);

    });
    $("#submit").click(function () {
        if (!$('#imgInput').hasClass('d-none')) {
            sendFile();
            $('.grid-show iframe').remove();

        }
    });
});

$(window).on('load', function () {
    let x = window.location.href.substring(window.location.href.lastIndexOf('/') + 1);
    console.log(x.toString());
    if (x == "") {
        $('#denoise').css({
            'background-color': 'rgba(208, 144, 144, 0.55)',
            'border-radius': '0.2em',
            'color': '#718daa'
        });

    } else
        $('#' + x.toString()).css({
            'background-color': 'rgba(208, 144, 144, 0.55)',
            'border-radius': '0.2em',
            'color': '#718daa'
        });
})