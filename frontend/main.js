const logWindow = document.querySelector("#log")
const dropArea = document.querySelector("#drop-area")
const dropAreaBtn = document.querySelector("#drop-area-button")
const dropAreaInput = document.querySelector("#drop-area-input")
const videoElem = document.querySelector("#video")
const videoCanvas = document.querySelector("#video-canvas")
// controls
const inputVideoSize = document.querySelector("#video-size")
const inputPreviewScores = document.querySelector("#preview-scores")
const inputTreshold = document.querySelector("#treshold")
const inputBackground = document.querySelector("#background")
const inputShape = document.querySelector("#shape")
// const videoCanvasCtx = videoCanvas.getContext("2d")
const downloadBtn = document.querySelector("#download-button")
const downloadProgressBar = document.querySelector("#video-download-progress")
const downloadProgressBarLabel = document.querySelector("#video-download-progress-label")

const labels_cache = {} // video labels will be stored here
const debug = false;
const accessedFileSlicesCanvas = document.querySelector("#accessed-file-slices-canvas")

dropArea.addEventListener("dragenter", (e) => { e.stopPropagation(); e.preventDefault(); }, false);
dropArea.addEventListener("dragover", (e) => { e.stopPropagation(); e.preventDefault(); }, false);
dropArea.addEventListener("drop", filesDropped, false);
dropAreaBtn.addEventListener("click", (e) => { if (dropAreaInput) dropAreaInput.click() }, false);
dropAreaInput.addEventListener("change", handleInputFieldFiles, false);
videoElem.addEventListener("loadeddata", setupPreview);

function initAccessedFileSclicesBar(){
    var dst = new cv.Mat(10, 1500, cv.CV_8UC4, new cv.Scalar(255, 0, 0, 255));
    cv.imshow('accessed-file-slices-canvas', dst);
}
function updateAccessedFileSclicesBar(beg, end, fsize){
    var src = cv.imread('accessed-file-slices-canvas');
    var width = src.size().width
    var height = src.size().height
    var roi = src.roi(new cv.Rect((width*beg/fsize), 0, (width*(end/fsize-beg/fsize)), height))
    var roi_new = new cv.Mat(roi.rows, roi.cols, roi.type(), new cv.Scalar(0, 255, 0, 255));
    roi_new.copyTo(roi)
    cv.imshow('accessed-file-slices-canvas', src);
}

function setupPreview() {
    videoElem.height = videoElem.videoHeight * inputVideoSize.value / 100
    videoElem.width = videoElem.videoWidth * inputVideoSize.value / 100
    const frame = new cv.Mat(videoElem.height, videoElem.width, cv.CV_8UC4);
    const cap = new cv.VideoCapture(videoElem);

    var lastTimestamp = -1; // videoElem.currentTime;
    function refreshFrame() {
        // this method should be called in cases like manual rewind, video resize event etc.
        if (lastTimestamp == videoElem.currentTime) return;
        lastTimestamp = videoElem.currentTime;
        processFrame();
    }
    function processFrame() {
        cap.read(frame);
        applyLabels(frame, getLabels(videoElem.currentTime), getConfig())
        cv.imshow(videoCanvas, frame);
    }
    function processVideo() {
        try {
            if (videoElem.ended) {
                return;
            }
            if (videoElem.paused) {
                requestAnimationFrame(processVideo);
                return;
            }
            if (lastTimestamp == videoElem.currentTime) {
                requestAnimationFrame(processVideo);
                return;
            }
            const transformFPS = 1 / (videoElem.currentTime - lastTimestamp)
            lastTimestamp = videoElem.currentTime;
            const begin = Date.now();
            processFrame();
            // schedule the next one.
            const delay = Date.now() - begin;
            logWindow.innerText = `compute_time due to on-the-fly preview: ${delay.toFixed(2)} ms \n 1/compute_time (limit FPS capacity): ${(1000 / delay).toFixed(2)} \npreview FPS ${transformFPS.toFixed(2)}`
            // setTimeout(processVideo, 1000/videoElem.FPS - delay);
            requestAnimationFrame(processVideo);
        } catch (err) {
            console.log("error captured: ", err);
        }
    };
    videoElem.onended = () => { lastTimestamp = -1; processFrame() }
    videoElem.onplay = () => requestAnimationFrame(processVideo);
    videoElem.onseeked = refreshFrame
    inputVideoSize.oninput = setupPreview

    inputPreviewScores.onchange = processFrame
    inputTreshold.onchange = processFrame
    inputBackground.onchange = processFrame
    inputShape.onchange = processFrame

    refreshFrame();
}

function filesDropped(e) {
    console.log('File(s) dropped');
    e.stopPropagation(); e.preventDefault();
    var files;
    if (e.dataTransfer.items) {
        // Use DataTransferItemList interface to access the file(s)
        files = [...e.dataTransfer.items].map((x) => x.getAsFile());
    } else {
        files = e.dataTransfer.files;
    }
    console.log(files);
    filesReady(files);
}
function handleInputFieldFiles(e) {
    // file coming from selector invoked by the dropAreaBtn button
    console.log("handleFiles");
    const files = e.target.files;
    console.log(files);
    if (files.length < 1) {
        console.log("no file present");
        return;
    }
    filesReady(files);
}
function filesReady(files) {
    console.log("filesReady", `files.length: ${files.length}`)
    if (files.length < 1) {
        alert("notice: file must be provided");
        return;
    }
    if (files.length > 1) {
        // TODO change alert to something else
        alert("notice: only the first file that you provided will be used");
    }

    console.log(files);
    file_pointer = files[0];
    // readable_stream = new ReadableStream();
    // videoElem.src = URL.createObjectURL(files[0]);
    // videoElem.hidden = false;
    // videoElem.srcObject = readable_stream;

    initAccessedFileSclicesBar();
    downloadBtn.disabled = true;
    downloadProgressBar.value = 0
    downloadProgressBar.hidden = true
    downloadProgressBarLabel.innerText = "estimating remaining time left..."
    downloadProgressBarLabel.hidden = true
    connectFcn(file_pointer)
    // videoElem.src = URL.createObjectURL(files[0]);
    // videoElem.hidden = false;
}

function connectFcn(file) {
    if (typeof ws == "object") {
        if (ws.readyState == ws.OPEN) {
            ws.close();
        }
    }
    logWindow.innerText += `\n connectFcn \n`
    Object.keys(labels_cache).map(function (key, idx) {
        delete labels_cache[key];
    });
    logWindow.innerText += `connectFcn1 \n`
    const ws_server = "ws://" + location.host + "/"
    ws = new WebSocket(ws_server + "ws");
    logWindow.innerText += `connectFcn2 \n`

    ws.addEventListener("open", () => {
        console.log("socket connected");
        var { lastModified, name, size, type } = file;
        var metadata = JSON.stringify({
            lastModified: lastModified,
            name: name,
            size: size,
            type: type,
            msg: "file available"
        });
        ws.send(metadata);
    })
    ws.addEventListener("close", () => console.log("socket disconnected"))
    ws.addEventListener("message", (msg) => {
        const data = JSON.parse(msg.data);
        // console.log("msg arrived", msg);
        console.log(`msg arrived ${data["msg"]}`);
        switch (data["msg"]) {
            case 'new file response':
                // datalink_ws.send(data["datalink name"]);
                videoElem.src = URL.createObjectURL(file);
                videoElem.hidden = false;
                videoElem.FPS = data["FPS"]; // add new property into videoElem 
                //videoElem.total_frames = data["total frames"]; // add new property into videoElem 
                downloadBtn.hidden = false;
                downloadBtn.disabled = false;
                downloadBtn.onclick = () => {
                    ws.send(JSON.stringify({ ...getConfig(), "msg": "user config, request download" }))
                    console.log("config send to the server, download will start when response arrives [insert loading GIF or smth]")
                }
                break;
            case 'lab':
                // console.log("############ labels arrived:", data["lab"]);
                const frame_index = Math.round(videoElem.currentTime * videoElem.FPS);
                const refresh = frame_index[frame_index] == undefined || frame_index[frame_index] == "awaiting";
                Object.assign(labels_cache, data["lab"]);
                if (refresh && frame_index[frame_index] != "awaiting") {
                    videoElem.currentTime += 0.0001; //trigger seek
                    // videoElem.play();
                }
                break;
            case 'get': //request to get slice of file
                const { S, E } = data;
                console.log("msg get:", S, E);
                fr = new FileReader;
                fr.requested_offset = S;
                fr.addEventListener('load', (e) => {
                    console.log("FileReader load event; sending the data");
                    console.log(e);
                    // ws.send(e.target.result);

                    offset_buffer = new ArrayBuffer(8);
                    var dataview = new DataView(offset_buffer);
                    // dataview.setBigInt64(0, BigInt(S)); // BigEndian by deafault
                    dataview.setBigInt64(0, BigInt(e.target.requested_offset)); // BigEndian by deafault
                    const data = new Blob([offset_buffer, e.target.result]);
                    ws.send(data);
                    updateAccessedFileSclicesBar(S,E, file.size)
                }, false);
                fr.readAsArrayBuffer(file.slice(S, E));
                break;
            case 'download ready':
                const a = document.createElement('a');
                a.href = data["path"];
                a.download = file.name;
                a.type = file.type;
                console.log("temp <a> node used to download the file:", a, file)
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                downloadProgressBar.hidden = false
                downloadProgressBarLabel.hidden = false
                downloadProgressBarLabel.innerText = "time left: estimating..."
                break;
            case 'progress':
                console.log("progress", data)
                const estimated_time_left = data['estimated_time_left'] // in seconds
                const estimated_time_left_string = new Date(estimated_time_left * 1000).toISOString().slice(11, 19);
                downloadProgressBar.value = data['ratio_done'] * 100
                downloadProgressBarLabel.innerText = "estimated time left: " + estimated_time_left_string
                break;
            default:
                console.log("incoming msg type not recognised", data)
        }
    })
}

function disconnectFcn() { // currently unused, TODO FLAG
    //this function is called if user changes the input file
    ws.close();
}

function applyLabels(frame, labels, config) {
    const T = config["treshold"];
    const shape = config["shape"];
    const background = config["background"];
    const previewScores = config["preview-scores"];
    if (labels == "awaiting" || labels == false) {
        cv.cvtColor(frame, frame, cv.COLOR_RGBA2GRAY);
        cv.cvtColor(frame, frame, cv.COLOR_GRAY2RGBA);
        if (videoElem.ended) {
            cv.putText(frame, "video ended", new cv.Point(0, 15),
                cv.FONT_HERSHEY_COMPLEX, 0.5, new cv.Scalar(10, 250, 10, 255));
            return;
        };
        if (ws.readyState == ws.OPEN) {
            cv.putText(frame, "Synchronizing...", new cv.Point(0, 15),
                cv.FONT_HERSHEY_COMPLEX, 0.5, new cv.Scalar(10, 250, 10, 255));
            cv.putText(frame, "consider pausing or slowing down the video", new cv.Point(0, 30),
                cv.FONT_HERSHEY_COMPLEX, 0.5, new cv.Scalar(10, 250, 10, 255));
        } else {
            cv.putText(frame, "Disconnected", new cv.Point(0, 15),
                cv.FONT_HERSHEY_COMPLEX, 0.5, new cv.Scalar(10, 250, 10, 255));
            cv.putText(frame, "consider contacting the website owner if message persist", new cv.Point(0, 30),
                cv.FONT_HERSHEY_COMPLEX, 0.5, new cv.Scalar(10, 250, 10, 255));
        }
    } else {
        for (const l of labels) if (l[0] >= T) applyLabel(frame, l, shape, background, previewScores);
    }
}
function applyLabel(frame, label, shape, background, previewScores) {
    const score = label[0]
    const scale = inputVideoSize.value / 100
    const [x0, y0, x1, y1] = label.slice(1, 5).map((v) => v * scale);
    const center = [(x1 + x0) / 2, (y1 + y0) / 2];
    const HW = [(x1 - x0), (y1 - y0)];
    const roi = frame.roi(new cv.Rect(y0, x0, y1 - y0, x1 - x0))

    switch (background) {
        case 'black':
            var roi_new = new cv.Mat(roi.rows, roi.cols, roi.type(), new cv.Scalar(0, 0, 0, 255));
            break;
        case 'blur':
            pseudo_pxls = 3;
            var roi_new = new cv.Mat()
            cv.blur(roi, roi_new, new cv.Size(HW[1] / pseudo_pxls, HW[0] / pseudo_pxls));
            break;
        case 'pixelate':
            pseudo_pxls = 3;
            var roi_new = new cv.Mat()
            cv.resize(roi, roi_new, new cv.Size(pseudo_pxls, pseudo_pxls), 0, 0, cv.INTER_LINEAR)
            cv.resize(roi_new, roi_new, roi.size(), 0, 0, cv.INTER_NEAREST)
            break;
    }

    switch (shape) {
        case 'ellipse':
            var ellipse = new cv.Mat.zeros(roi.rows, roi.cols, cv.CV_8U);
            cv.ellipse(ellipse, new cv.Point(HW[1] / 2, HW[0] / 2), new cv.Size(HW[1] / 2, HW[0] / 2),
                0, 0, 360,
                new cv.Scalar(1), //any non-zero value would go
                -1)
            roi_new.copyTo(roi, ellipse)
            break;
        case 'rectangle':
            roi_new.copyTo(roi)
            break;
        case 'bbox':
            const s = frame.size()
            const S = Math.max(1, 0.0016 * Math.max(s["width"], s["height"]))
            cv.rectangle(frame, new cv.Point(y0, x0), new cv.Point(y1, x1), new cv.Scalar(10, 250, 10, 255), S)
            break;
    }
    if (previewScores) {
        const s = frame.size()
        const S = Math.max(1, 0.001 * Math.max(s["width"], s["height"]))
        cv.putText(frame, `${score.toFixed(2)}`, new cv.Point(y0, x0 - 1),
            cv.FONT_HERSHEY_PLAIN, S, new cv.Scalar(10, 250, 10, 255))
    }
}
function getLabels(timestamp) {
    const frame_index = Math.round(videoElem.currentTime * videoElem.FPS);
    console.log(`getLabels, frame index: ${frame_index}`)
    // decide if there is need to pull next batch of labels
    const buffer_in_seconds = 2; // edit here as needed
    // const buffer_in_seconds = window.default_config["client_label_buffer_in_seconds"];
    // ^^ consider making measurements at initial connection and informing the client what buffer to use
    const B = Math.round(buffer_in_seconds * videoElem.FPS);
    // check if there are next B labels; request new labels if not
    var need_for_labels = false;
    var i = frame_index;
    for (; i < frame_index + B; ++i) {
        //if (!labels_cache.hasOwnProperty(i)) {
        if (labels_cache[i] == undefined) {
            need_for_labels = true;
            break;
        }
    }
    if (need_for_labels) {
        //request labels from the first unavailable frame-label up to 2*buffer_in_seconds ahead from current position
        const frame_2B_ahead = frame_index + 2 * B;
        //const j = Math.min(videoElem.total_frames, frame_2B_ahead);
        const msg = JSON.stringify({ "msg": "get", "from": i, "upto": frame_2B_ahead })
        console.log(`asking for labels ${msg}`)
        ws.send(msg);
        // ws.send(JSON.stringify({"msg":"get", "req":`${i}-${frame_2B_ahead}`}));
        // for (var j=i; j<frame_2B_ahead;++j){
        //     labels_cache[j] = "awaiting"
        // }
    }
    if (labels_cache.hasOwnProperty(frame_index)) {
        const labels = labels_cache[frame_index];
        // if (labels == "awaiting") videoElem.pause();
        return labels;
    } else {
        videoElem.pause();
        return false; // signal that no labels are available atm
    }
}
function getConfig() {
    return {
        "treshold": inputTreshold.value / 100,
        "shape": inputShape.value,
        "background": inputBackground.value,
        "preview-scores": inputPreviewScores.checked,
    };
}
function setConfig(config) {
    // after websocket initialization server should send default config
    inputTreshold.value = config["treshold"] * 100;
    inputShape.value = config["shape"];
    inputBackground.value = config["background"];
    inputPreviewScores.checked = config["preview-scores"];
    window.default_pipeline_config = config; // extra varuables could go here
};

// TODO video slow down/speed up buttons
