
// Arrays used for machine learning
let labels = [];
let colors = [];

let labelList = [
    'red-ish',
    'green-ish',
    'blue-ish',
    'orange-ish',
    'yellow-ish',
    'pink-ish',
    'purple-ish',
    'brown-ish',
    'grey-ish'
]

// Array used for visualisation
let colorByLabel = {
    'red-ish': [],
    'green-ish': [],
    'blue-ish': [],
    'orange-ish': [],
    'yellow-ish': [],
    'pink-ish': [],
    'purple-ish': [],
    'brown-ish': [],
    'grey-ish': []
}

function setup() {
    createCanvas(100, displayHeight);

    // Initialise Firebase
    const config = {
        apiKey: "AIzaSyAMNh1LmtxB-19rGReXct1QnE6ZrNNsaXw",
        authDomain: "color-label-dataset.firebaseapp.com",
        databaseURL: "https://color-label-dataset.firebaseio.com",
        projectId: "color-label-dataset",
        storageBucket: "",
        messagingSenderId: "365128804562"
    }
    firebase.initializeApp(config);
    database = firebase.database();

    // Load color label data from Firebase
    let ref = database.ref('colors');
    ref.once('value', (result) => {
        let data = result.val();

        // Process data
        for (let key of Object.keys(data)) {
            let record = data[key]
            let col = color(record.r, record.g, record.b);
            colorByLabel[record.label].push(col);
            let colNormalised = [record.r / 255, record.g / 255, record.b / 255];
            colors.push(colNormalised);
            labels.push(labelList.indexOf(record.label));
        }

        // Visualise all color data
        let x = 0;
        let y = 0;

        for (let label of Object.keys(colorByLabel)) {
            console.log(`${label} ${colorByLabel[label].length}`);

            for (let i = 0; i < colorByLabel[label].length; i++) {
                noStroke();
                fill(colorByLabel[label][i]);
                rect(x, y, 10, 10);
                x += 10;
                if (x >= width) {
                    x = 0;
                    y += 10;
                }
            }

            y += 20;
            x = 0;
        }

        // Create tensors
        let xs = tf.tensor2d(colors);
        let ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 9);
        xs.print();
        console.log(xs.shape);
        ys.print();
        console.log(ys.shape);

        // Create model
        const model = tf.sequential({
            layers: [
                tf.layers.dense({ // hidden layer
                    units: 16, // number of nodes
                    activation: 'sigmoid',
                    inputDim: [3] // input shape (can also be {{inputShape}})
                }),
                // Output layer
                tf.layers.dense({ //output layer
                    units: 16,
                    activation: 'sigmoid' //'relu''softmax''hardSigmoid''elu'
                })
            ]
        });


        // Hyper parameters to fine-tune the output
        const learningRate = 0.01;

        // Train model
        const opt = tf.train.sgd(learningRate); //optimizer using gradient descent

        model.compile({
            optimizer: opt,
            loss: 'meanSquaredError' // loss function
        });

        for (i = 0; i < 10; i++) {
            const h = model.fit(xs, ys, {
                batchSize: 32,
                eposchs: 10
            })
            .then((h) => console.log(h));
        }



    });
}