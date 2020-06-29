import Vue from 'vue';
import HTTP from './http';
import axios from 'axios';

const vm = new Vue({
    el: '#mail-app',
    delimiters: ['[[', ']]'],
    data () {
        return {
            annotators: [],
        }
    },
    methods: {
        updateUsers() {
            console.log('Called from updateUsers');

            var i=0;
            for (; i<this.annotators.length; i++) {
                const payload = {
                    user_id: this.annotators[i].id,
                    username: this.annotators[i].username,
                    user_role: this.annotators[i].user_role,
                    doc_per_session: this.annotators[i].doc_per_session,
                };
                console.log(payload);
                axios.patch('/api/profiles/' + this.annotators[i].id + '/', payload).then((response) => {});
            }
        },
        reset() {

        }
    },
    created() {
        console.log('Called from created');
        HTTP.get('users').then((response) => {
            this.annotators = response.data;
        });
    },
});