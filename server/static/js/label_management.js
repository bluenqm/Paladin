import Vue from 'vue';
import HTTP from './http';

const vm = new Vue({
    el: '#mail-app',
    delimiters: ['[[', ']]'],
    data: {
        labels: [],
        labelText: '',
        selectedKey: '',
        checkedKey: [],
        shortcutKey: '',
        backgroundColor: '#209cee',
        textColor: '#ffffff',
        error_message: '',
    },

    computed: {
        /**
         * combineKeys: Combine selectedKey and checkedKey to get shortcutKey
         * saveKeys: Save null to database if shortcutKey is empty string
         */
        combineKeys: function() {
            this.shortcutKey = '';

            // If checkedKey exits, add it to shortcutKey
            if (this.checkedKey.length > 0) {
                this.checkedKey.sort();
                this.shortcutKey = this.checkedKey.join(' ');

                // If selectedKey exist, add it to shortcutKey
                if (this.selectedKey.length !== 0) {
                    this.shortcutKey = this.shortcutKey + ' ' + this.selectedKey;
                }
            }

            // If only selectedKey exist, assign to shortcutKey
            if (this.shortcutKey.length === 0 && this.selectedKey.length !== 0) {
                this.shortcutKey = this.selectedKey;
            }
            return this.shortcutKey;
        },

        saveKeys: function() {
            this.shortcutKey = this.combineKeys;
            if (this.shortcutKey === '') {
                return null;
            }
            return this.shortcutKey;
        },
    },

    methods: {
        addLabel() {
            const payload = {
                text: this.labelText,
                shortcut: this.saveKeys,
                background_color: this.backgroundColor,
                text_color: this.textColor,
            };

            HTTP.post('labels/', payload)
            .then((response) => {
                this.reset();
                this.labels.push(response.data);
            })
            .catch(error => {
                this.error_message = error.response.statusText;
            });
        },

        removeLabel(label) {
            const labelId = label.id;
            HTTP.delete(`labels/${labelId}`).then((response) => {
                const index = this.labels.indexOf(label);
                this.labels.splice(index, 1);
            });
        },

        reset() {
            this.labelText = '';
            this.selectedKey = '';
            this.checkedKey = [];
            this.shortcutKey = '';
            this.backgroundColor = '#209cee';
            this.textColor = '#ffffff';
            this.error_message = '';
        },
    },

    created() {
        HTTP.get('labels').then((response) => {
            this.labels = response.data;
        });
    },
});