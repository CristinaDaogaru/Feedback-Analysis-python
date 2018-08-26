using System;
using System.Globalization;
using System.Linq;
using System.Security.Claims;
using System.Threading.Tasks;
using System.Web;
using System.Web.Mvc;
using FeedbackPlatform.Models;
using System.Web.Security;
using FeedbackPlatform.Database;
using FeedbackPlatform.Common;
using System.Data.Entity.Validation;

namespace FeedbackPlatform.Controllers
{
    

    [Authorize]
    public class AccountController : Controller
    {
        #region Added by me
        public ActionResult Index()
        {
            return View();
        }

        [HttpGet]
        [AllowAnonymous]
        public ActionResult Login()
        {
            return View();
        }

        [HttpPost]
        [AllowAnonymous]
        [ValidateAntiForgeryToken]
        public async Task<ActionResult> Login(ClientProfile userr)
        {
            var db = new DatabaseContext();
            var user = db.ClientProfiles.FirstOrDefault(u => u.Email == userr.Email);

            
            if (IsValid(userr.Email, userr.Password))
            {
                return RedirectToAction("Home", "Home", new { clientId = user.IdClient });
            }
            else
            {
                ModelState.AddModelError("", "Login details are wrong.");
            }
            ViewBag.id = user.IdClient;
            return View(userr);
        }
        [HttpGet]
        [AllowAnonymous]
        public ActionResult Register()
        {
            return View();
        }

        [HttpPost]
        [AllowAnonymous]
        [ValidateAntiForgeryToken]
        public async Task<ActionResult> Register(ClientProfile user)
        {
            try
            {
                if (ModelState.IsValid)
                {
                    using (var db = new FeedbackPlatform.Database.DatabaseContext())
                    {
                        var crypto = new PasswordEncode();
                        var encrypPass = crypto.Hash(user.Password);
                        var newUser = db.ClientProfiles.Create();
                        newUser.Email = user.Email;
                        newUser.Password = encrypPass;
                        newUser.Name = user.Name;
                        db.ClientProfiles.Add(newUser);
                        db.SaveChanges();
                        return RedirectToAction("Index", "Home");
                    }
                }
                else
                {
                    ModelState.AddModelError("", "Data is not correct");
                }
            }
            catch (DbEntityValidationException e)
            {
                foreach (var eve in e.EntityValidationErrors)
                {
                    Console.WriteLine("Entity of type \"{0}\" in state \"{1}\" has the following validation errors:",
                        eve.Entry.Entity.GetType().Name, eve.Entry.State);
                    foreach (var ve in eve.ValidationErrors)
                    {
                        Console.WriteLine("- Property: \"{0}\", Error: \"{1}\"",
                            ve.PropertyName, ve.ErrorMessage);
                    }
                }
                throw;
            }
            return View();
        }

        public ActionResult LogOut()
        {
            FormsAuthentication.SignOut();
            return RedirectToAction("Index", "Home");
        }

        private bool IsValid(string email, string password)
        {
            var crypto = new PasswordEncode();
            bool IsValid = false;

            using (var db = new FeedbackPlatform.Database.DatabaseContext())
            {
                var user = db.ClientProfiles.FirstOrDefault(u => u.Email == email);
                if (user != null)
                {
                    if (user.Password == crypto.Hash(password))
                    {
                        IsValid = true;
                    }
                }
            }
            return IsValid;
        }
        #endregion
    }

}
