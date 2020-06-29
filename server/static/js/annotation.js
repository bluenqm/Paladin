import Vue from 'vue';
import HTTP from './http';

Vue.use(require('vue-shortkey'), {
    prevent: ['input', 'textarea'],
});

const vm = new Vue({
    el: '#mail-app',
    data() {
        return {
            labels: [],
            docs: [],
            annotations: [],
            doc_index: 0,
        }
    },
    delimiters: ['[[', ']]'],
    created() {
        console.log('Called from created');
        /* Get docs and annotations from API */
        HTTP.get('docs').then((response) => {
            this.docs = response.data;
            var all_annotations = this.docs.map(doc => doc.annotations);
            HTTP.get('annotations').then((response) => {
                var filtered_annotations = response.data;
                for (var i = 0; i < all_annotations.length; i++) {
                    all_annotations[i] = all_annotations[i].filter(function(ele) {
                        for (var j = 0; j<filtered_annotations.length; j++) {
                            if (filtered_annotations[j].id == ele.id)
                                return true;
                        }
                        return false;
                    });
                }
                this.annotations = all_annotations;
            });
        });

        /* Get labels from API */
        HTTP.get('labels').then((response) => {
            this.labels = response.data;
        });
    },
    methods: {
        async addLabel(label) {
            console.log('Called from addLabel');
            const annotation = this.getAnnotation(label);
            if (annotation) {
                this.removeLabel(annotation);
            }
            else {
                const docId = this.docs[this.doc_index].id;
                const payload = {
                    label: label.id,
                };
                const query = 'docs/' + docId + '/annotations/';
                await HTTP.post(query, payload).then((response) => {
                    this.annotations[this.doc_index].push(response.data);
                });
            }
        },

        removeLabel(annotation) {
            console.log('Called from removeLabel');
            const docId = this.docs[this.doc_index].id;
            const query = 'docs/' + docId + '/annotations/' + annotation.id;
            HTTP.delete(query).then(() => {
                const index = this.annotations[this.doc_index].indexOf(annotation);
                this.annotations[this.doc_index].splice(index, 1);
            });
        },

        getAnnotation(label) {
            console.log('Called from getAnnotation');
            const currentAnnotation = this.annotations[this.doc_index];
            if (currentAnnotation == null) {
                return [];
            }

            const found = currentAnnotation.find(function(element) {
                return element.label == label.id;
            });
            return found;
        },

        async nextPage() {
            this.doc_index += 1;
            if (this.doc_index == this.docs.length) {
                this.doc_index = this.docs.length - 1;
            }
        },

        async prevPage() {
            this.doc_index -= 1;
            if (this.doc_index == -1) {
                this.doc_index = 0;
            }
        },

        async newBatch() {
            HTTP.get('proactive').then((response) => {
                console.log("Finish Annotation");
                const baseUrl = window.location.href.split('/').slice(0, 4).join('/');
                window.location.href = baseUrl;
            });
        },

        replaceNull(shortcut) {
            if (shortcut === null) {
                shortcut = '';
            }
            shortcut = shortcut.split(' ');
            return shortcut;
        },
    },

    computed: {
        id2label() {
            const id2label = {};
            for (let i = 0; i < this.labels.length; i++) {
                const label = this.labels[i];
                id2label[label.id] = label;
            }
            return id2label;
        },
    }
});