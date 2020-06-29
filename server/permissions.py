from django.contrib.auth.mixins import UserPassesTestMixin


class SuperUserMixin(UserPassesTestMixin):
    def test_func(self):
        return self.request.user.is_superuser