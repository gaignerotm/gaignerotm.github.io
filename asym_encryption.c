#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Implémentation de l'algorithme RSA.
// Nécessite la bibliothèque de programmation GMP, car les nombres sont manipulés avec GMP.
// Commande de compilation:gcc asym_encryption.c -lgmp -o asym_encryption

void print_usage(const char *prog) {
    printf("Usage :\n");
    printf("  %s -g -p <p> -q <q> -k <e>\n", prog);
    printf("  %s -e -m <message> -n <n> -k <e>\n", prog);
    printf("  %s -d -m <cipher>  -n <n> -k <d>\n", prog);
    printf("\n");
    printf("Exemple generation : %s -g -p 61 -q 53 -k 17\n", prog);
    printf("Exemple chiffrement : %s -e -m 65 -n 3233 -k 17\n", prog);
    printf("Exemple dechiffrement : %s -d -m 2790 -n 3233 -k 2753\n", prog);
}

/* Vérifie que e peut être utilisé avec phi(n). */
int public_exponent_is_valid(const mpz_t e, const mpz_t phi) {
    mpz_t gcd;
    mpz_init(gcd);
    mpz_gcd(gcd, e, phi);
    int ok = (mpz_cmp_ui(e, 1) > 0) && (mpz_cmp(e, phi) < 0) && (mpz_cmp_ui(gcd, 1) == 0);
    mpz_clear(gcd);
    return ok;
}

/* Calcule n, phi et d à partir de p, q et e. */
int generate_keys(mpz_t n, mpz_t phi, mpz_t d, const mpz_t p, const mpz_t q, const mpz_t e) {
    mpz_t p1, q1;
    mpz_inits(p1, q1, NULL);

    mpz_mul(n, p, q);
    mpz_sub_ui(p1, p, 1);
    mpz_sub_ui(q1, q, 1);
    mpz_mul(phi, p1, q1);

    if (!public_exponent_is_valid(e, phi)) {
        mpz_clears(p1, q1, NULL);
        return 0;
    }

    if (mpz_invert(d, e, phi) == 0) {
        mpz_clears(p1, q1, NULL);
        return 0;
    }

    mpz_clears(p1, q1, NULL);
    return 1;
}

/* Chiffrement RSA : c = m^e mod n */
int rsa_encrypt(mpz_t cipher, const mpz_t message, const mpz_t e, const mpz_t n) {
    if (mpz_cmp(message, n) >= 0 || mpz_sgn(message) < 0) {
        return 0;
    }
    mpz_powm(cipher, message, e, n);
    return 1;
}

/* Déchiffrement RSA : m = c^d mod n */
void rsa_decrypt(mpz_t message, const mpz_t cipher, const mpz_t d, const mpz_t n) {
    mpz_powm(message, cipher, d, n);
}

int read_mpz_argument(mpz_t out, const char *value) {
    return mpz_set_str(out, value, 0) == 0;
}

int main(int argc, char *argv[]) {
    int mode_generate = 0;
    int mode_encrypt = 0;
    int mode_decrypt = 0;

    const char *arg_p = NULL;
    const char *arg_q = NULL;
    const char *arg_m = NULL;
    const char *arg_n = NULL;
    const char *arg_k = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-g") == 0) {
            mode_generate = 1;
        } else if (strcmp(argv[i], "-e") == 0) {
            mode_encrypt = 1;
        } else if (strcmp(argv[i], "-d") == 0) {
            mode_decrypt = 1;
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            arg_p = argv[++i];
        } else if (strcmp(argv[i], "-q") == 0 && i + 1 < argc) {
            arg_q = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            arg_m = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            arg_n = argv[++i];
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            arg_k = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (mode_generate + mode_encrypt + mode_decrypt != 1) {
        print_usage(argv[0]);
        return 1;
    }

    mpz_t p, q, m, n, k, phi, d, result;
    mpz_inits(p, q, m, n, k, phi, d, result, NULL);

    if (mode_generate) {
        if (!arg_p || !arg_q || !arg_k) {
            print_usage(argv[0]);
            mpz_clears(p, q, m, n, k, phi, d, result, NULL);
            return 1;
        }

        if (!read_mpz_argument(p, arg_p) || !read_mpz_argument(q, arg_q) || !read_mpz_argument(k, arg_k)) {
            printf("Erreur : argument invalide.\n");
            mpz_clears(p, q, m, n, k, phi, d, result, NULL);
            return 1;
        }

        if (mpz_probab_prime_p(p, 20) == 0 || mpz_probab_prime_p(q, 20) == 0) {
            printf("Erreur : p et q doivent etre premiers.\n");
            mpz_clears(p, q, m, n, k, phi, d, result, NULL);
            return 1;
        }

        if (!generate_keys(n, phi, d, p, q, k)) {
            printf("Erreur : e n'est pas compatible avec phi(n).\n");
            mpz_clears(p, q, m, n, k, phi, d, result, NULL);
            return 1;
        }

        gmp_printf("n  = %Zd\n", n);
        gmp_printf("phi= %Zd\n", phi);
        gmp_printf("e  = %Zd\n", k);
        gmp_printf("d  = %Zd\n", d);
    }

    if (mode_encrypt) {
        if (!arg_m || !arg_n || !arg_k) {
            print_usage(argv[0]);
            mpz_clears(p, q, m, n, k, phi, d, result, NULL);
            return 1;
        }

        if (!read_mpz_argument(m, arg_m) || !read_mpz_argument(n, arg_n) || !read_mpz_argument(k, arg_k)) {
            printf("Erreur : argument invalide.\n");
            mpz_clears(p, q, m, n, k, phi, d, result, NULL);
            return 1;
        }

        if (!rsa_encrypt(result, m, k, n)) {
            printf("Erreur : le message doit verifier 0 <= m < n.\n");
            mpz_clears(p, q, m, n, k, phi, d, result, NULL);
            return 1;
        }

        gmp_printf("cipher = %Zd\n", result);
    }

    if (mode_decrypt) {
        if (!arg_m || !arg_n || !arg_k) {
            print_usage(argv[0]);
            mpz_clears(p, q, m, n, k, phi, d, result, NULL);
            return 1;
        }

        if (!read_mpz_argument(m, arg_m) || !read_mpz_argument(n, arg_n) || !read_mpz_argument(k, arg_k)) {
            printf("Erreur : argument invalide.\n");
            mpz_clears(p, q, m, n, k, phi, d, result, NULL);
            return 1;
        }

        rsa_decrypt(result, m, k, n);
        gmp_printf("message = %Zd\n", result);
    }

    mpz_clears(p, q, m, n, k, phi, d, result, NULL);
    return 0;
}
