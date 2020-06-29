const process = require('process');
const VueLoaderPlugin = require('vue-loader/lib/plugin');

module.exports = {
    mode: process.env.DEBUG === 'False' ? 'production' : 'development',
    entry: {
        'projects': './static/js/projects.js',
        'annotation': './static/js/annotation.js',
        'label_management': './static/js/label_management.js',
        'stats': './static/js/stats.js',
        'setting': './static/js/setting.js',
        'upload': './static/js/upload.js',
        'profile': './static/js/profile.js',
    },
    output: {
        path: __dirname + '/static/bundle',
        filename: '[name].js'
    },
    module: {
        rules: [
            {
                test: /\.vue$/,
                loader: 'vue-loader'
            },
            {
              test: /\.css$/,
              include: /node_modules/,
              loaders: ['style-loader', 'css-loader'],
             }
        ]
    },
    plugins: [
        new VueLoaderPlugin()
    ],
    resolve: {
        extensions: ['.js', '.vue'],
        alias: {
            vue$: 'vue/dist/vue.esm.js',
        },
    },
}